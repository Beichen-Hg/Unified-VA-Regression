"""Actual frozen training loops with a tiny local backbone; no model downloads."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy,json,sys,tempfile
import torch
from torch.utils.checkpoint import checkpoint
from unified_va.config import template,TARGETS,ORIGINAL_TRAIN,REVISION
from unified_va.source import load_engine
from unified_va.io import write,read
from unified_va.storage import overlay,save,restore
from unified_va import storage as portable_storage
from unified_va.train import Hooks
from unified_va.inference import reporting_predict


class TinyConfig(SimpleNamespace):
    def save_pretrained(self,folder):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        (folder/'config.json').write_text(json.dumps(vars(self)))


class Block(torch.nn.Module):
    def __init__(self):super().__init__();self.proj=torch.nn.Linear(8,8)
    def forward(self,hidden_states,attention_mask=None,**kwargs):return hidden_states+.03*torch.tanh(self.proj(hidden_states))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=TinyConfig(hidden_size=8,num_hidden_layers=36,pad_token_id=0,use_cache=False,_commit_hash=REVISION,model_type='qwen3')
        self.embed_tokens=torch.nn.Embedding(64,8);self.layers=torch.nn.ModuleList([Block() for _ in range(36)])
        self.norm=torch.nn.LayerNorm(8);self.checkpointing=False
    def gradient_checkpointing_enable(self,**kwargs):self.checkpointing=True
    def enable_input_require_grads(self):
        self.embed_tokens.register_forward_hook(lambda module,args,output:output.requires_grad_(True))
    def get_input_embeddings(self):return self.embed_tokens
    def forward(self,input_ids,attention_mask,use_cache=False,return_dict=True,**kwargs):
        value=self.embed_tokens(input_ids)
        for layer in self.layers:
            value=checkpoint(lambda x,layer=layer:layer(x,attention_mask=attention_mask),value,use_reentrant=True) if self.training and self.checkpointing else layer(value,attention_mask=attention_mask)
        return SimpleNamespace(last_hidden_state=self.norm(value))


class Tokenizer:
    pad_token='[PAD]';pad_token_id=0;eos_token='[EOS]';eos_token_id=2;unk_token='[UNK]'
    init_kwargs={'_commit_hash':REVISION}
    def __call__(self,text,**kwargs):
        length=kwargs.get('max_length',16);values=[ord(c)%32+1 for c in text][:length]
        return dict(input_ids=torch.tensor([values+[0]*(length-len(values))]),attention_mask=torch.tensor([[1]*len(values)+[0]*(length-len(values))]))
    def save_pretrained(self,folder):Path(folder).mkdir(parents=True,exist_ok=True)


def row(target,index=0):
    language,domain=target.split('_',1)
    return dict(id=target+str(index),text='Good service with a mixed response '+str(index),aspect='service',aspect_index=0,sample_aspects=['service'],
        lang=language,domain=domain,v=4.+index,a=5.+.5*index)


def run(role):
    torch.set_num_threads(2);engine=load_engine(role)
    import src.models.small_llm_regressor as module
    torch.manual_seed(0);base=Tiny()
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);cfg=template(role,'original')
        cfg.update(device='cpu',amp=False,n_epochs=1 if role=='stage2' else 3,grad_accum=5 if role=='stage2' else 1,max_len=16,
            eval_every_updates=2 if role=='stage2' else 5,output_dir=str(root/'outputs'))
        cfg['small_llm'].update(model_name=str(root/'local-base'),torch_dtype='float32')
        cfg['small_llm']['prompt_template']='Text: {text} Aspect: {aspect}'
        cfg['experiment'].update(run_dir=str(root),training_targets=list(ORIGINAL_TRAIN),data_root=str(root),model_dir=str(root/'local-base'))
        for source in cfg['auxiliary_data']['sources']:
            source['data_dir']=str(root/'official'/source['lang'])
        (root/'local-base').mkdir()
        training=[row(t,i%2) for i,t in enumerate(ORIGINAL_TRAIN)] if role=='stage2' else [row('jpn_hotel',i) for i in [0,1]]
        dev_targets=[dict(name=t,dev_data=[row(t,i) for i in [0,1]],eval_cfg=cfg) for t in ORIGINAL_TRAIN]
        primary=[row('jpn_hotel',i) for i in [0,1]]
        norm=engine._build_va_norm_stats(training,True);write(root/'DATA_AUDIT.json',dict(passed=True,normalization=norm))
        cfg['experiment'].update(train_aspects=len(training),dev_aspects=24,expected_updates=3 if role=='stage2' else 6,
            development_events=[2,3] if role=='stage2' else [2,4,5,6])
        if role=='stage2':
            cfg['matched_protocol'].update(train_only_normalization=norm,expected_training_aspects=12,updates_per_epoch=3,
                expected_total_updates=3,effective_batch_size_world1=5)
            cfg['method_validation']['update_observer']['enabled']=False
            cfg['pretrain_from']=None
            # Parentless fixture uses a fresh S-shaped model with the same tiny base.
            pcfg=copy.deepcopy(cfg);pcfg['backbone_conditional_adapter']['enabled']=False;pcfg['text_only_inference']=False
            pcfg['matched_protocol'].update(arm='S',protocol_version='recovered_original_stage1_source_20261104')
            pcfg['experiment']['stage']='stage1'
        else:pcfg=None
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100*2**30)),\
             patch.object(module.AutoModel,'from_pretrained',side_effect=lambda *a,**k:copy.deepcopy(base)),patch.object(module.AutoTokenizer,'from_pretrained',return_value=Tokenizer()),\
             patch.object(engine,'load_splits',return_value=dict(train=[],dev=primary,test=[])),\
             patch.object(engine,'load_samples',side_effect=lambda config,name:[] if name=='train' else primary),\
             patch.object(engine,'expand_to_aspect_level',side_effect=lambda rows:rows),\
             patch.object(engine,'_merge_auxiliary_data',return_value=training),patch.object(engine,'_build_macro_dev_targets',return_value=dev_targets):
            if role=='stage2':
                from src.train.checkpoint_storage import initialize_checkpoint_storage
                parent,_,_=engine.build_small_llm_regressor(pcfg);engine.freeze_small_llm_layers(parent,24)
                parent.matched_va_norm_stats=norm
                initialize_checkpoint_storage(parent,pcfg)
                parent_dir=root/'parent';parent_dir.mkdir();parent_file=parent_dir/'checkpoint.pt'
                # Use the source writer to create a genuine matched-parent storage payload.
                engine._save_checkpoint(parent,parent_dir,0,{},pcfg,global_update=2)
                native=torch.load(parent_file,map_location='cpu',weights_only=False)
                # Synthetic completed-S contract exercises the identical-tensor public bridge.
                # This fixture is not a paper model or a performance experiment.
                stage1=overlay(parent,pcfg,0,2,{},norm);save(parent_dir/'stage1.pt',stage1)
                write(parent_dir/'RESULT.json',dict(passed=True,stage='stage1',checkpoint='stage1.pt',updates=3,development_checks=2,selected_update=2))
                write(parent_dir/'FINAL_DEV_RESTORE.json',dict(passed=True))
                cfg['pretrain_from']=str(root/'checkpoints/shared_stage1.pt')
                portable_storage.bridge(parent_dir,cfg)
                converted=torch.load(cfg['pretrain_from'],map_location='cpu',weights_only=False)
                for field in ['backbone','head','pool_gate']:
                    assert set(native[field])==set(converted[field])
                    assert all(torch.equal(value,converted[field][key]) for key,value in native[field].items())
                from unified_va.inference import Predictor
                plain=Predictor(cfg['pretrain_from'],root/'local-base','cpu')
                assert plain.stage=='stage1' and len(plain.predict(primary))==2
                del plain
                del parent,native,converted
            hooks=Hooks(engine,cfg);hooks.install()
            try:
                metrics=engine.train_small_llm_va(cfg)
                result=hooks.finish(metrics)
            finally:hooks.close()
            assert result['passed'] and result['updates']==cfg['experiment']['expected_updates']
            assert read(root/'FINAL_DEV_RESTORE.json')['passed']
            gradients=read(root/'FIRST_OPTIMIZER_GRADIENTS.json')
            assert gradients['upper']['gradient_tensors']>0 if role=='stage2' else gradients['upper']['gradient_tensors']==0
            model,tokenizer,_=engine.build_small_llm_regressor(cfg);engine.freeze_small_llm_layers(model,24)
            if role!='stage2':restore(model,root/result['checkpoint'],cfg)
            else:
                aux=engine._setup_soft_condition_label_supervision(cfg,model,torch.device('cpu'))
                engine._load_checkpoint(model,root/result['checkpoint'],cfg=cfg,load_head='always',extra_modules={'soft_condition_label_head':aux})
            predictor=reporting_predict(engine,role)
            output=predictor(model,primary,tokenizer,torch.device('cpu'),cfg,va_norm_stats=norm)
            assert len(output)==2
            if role!='stage2':
                # Legacy S files used a generic top-level type with a different storage tag.
                from unified_va.inference import Predictor
                legacy=torch.load(root/result['checkpoint'],map_location='cpu',weights_only=False)
                legacy['checkpoint_type']='pinned_base_overlay'
                legacy['checkpoint_storage']['format']='absolute_tensor_overlay_v1'
                save(root/'legacy.pt',legacy)
                loaded=Predictor(root/'legacy.pt',root/'local-base','cpu')
                assert len(loaded.predict(primary))==2
            if role=='stage2':
                class HalfCent(torch.nn.Module):
                    def __init__(self):super().__init__();self.head=torch.nn.Linear(1,2)
                    def forward(self,input_ids,attention_mask,**kwargs):return torch.tensor([[6.483624458312988,6.045000076293945]])
                precision=copy.deepcopy(cfg);precision['text_only_inference']=False
                native=predictor(HalfCent(),primary[:1],tokenizer,torch.device('cpu'),precision)[0]['a_pred']
                development=engine._predict(HalfCent(),primary[:1],tokenizer,torch.device('cpu'),precision)[0]['a_pred']
                assert round(native,2)==6.04 and development==6.05
            print(json.dumps(dict(passed=True,role=role,updates=result['updates'],selected_update=result['selected_update'],tiny_source_training_only=True)))


if __name__=='__main__':run(sys.argv[1])
