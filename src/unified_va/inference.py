"""Text/aspect-only inference with post-forward association of IDs and gold labels."""
from pathlib import Path
import copy,inspect
from .source import load_engine
from . import storage


def reporting_predict(engine,stage):
    if stage!='stage2':return engine._predict
    text=inspect.getsource(engine._predict)
    old='va = torch.tensor([[round(float(x), 2) for x in row] for row in va], dtype=torch.float64)'
    if text.count(old)!=1:raise RuntimeError('Unknown source prediction precision path')
    text=text.replace(old,'va = (va * 100).round() / 100')
    namespace={};exec(compile(text,'<Unified VA regression-native-Torch-reporting>','exec'),engine.__dict__,namespace)
    return namespace['_predict']


def input_rows(rows):
    result=[]
    for i,row in enumerate(rows):
        if not isinstance(row.get('text'),str) or not isinstance(row.get('aspect'),str):raise ValueError('Each input needs string text and aspect fields')
        # No language, domain, source IDs, dataset identity, or gold crosses this boundary.
        result.append(dict(text=row['text'],aspect=row['aspect'],id=str(i),aspect_index=0,sample_aspects=[row['aspect']]))
    return result


def associate(predictions,rows):
    if len(predictions)!=len(rows):raise RuntimeError('Prediction count differs from inputs')
    result=[]
    for i,(prediction,row) in enumerate(zip(predictions,rows)):
        current=dict(id=row.get('id',str(i)),aspect=row['aspect'],aspect_index=int(row.get('aspect_index',0)),
            v_pred=prediction['v_pred'],a_pred=prediction['a_pred'])
        for label,key in [('v','v_true'),('a','a_true')]:
            if row.get(label) is not None:current[key]=float(row[label])
        if row.get('lang') and row.get('domain'):current['target']=row['lang']+'_'+row['domain']
        result.append(current)
    return result


class Predictor:
    def __init__(self,checkpoint,model_dir,device='cuda:0'):
        import torch
        self.device=torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
        payload=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        self.cfg=copy.deepcopy(payload['effective_inference_config'])
        self.stage=self.cfg.get('experiment',{}).get('stage') or ('stage2' if self.cfg.get('text_only_inference') else 'baseline' if self.cfg['head_type']=='linear' else 'stage1')
        # Existing S bridges use the audited matched-overlay reader, with no adapters.
        # This affects loading only; Stage1 training still uses its original engine.
        matched_plain=self.stage!='stage2' and payload.get('checkpoint_storage',{}).get('format')=='pinned_base_overlay_v1'
        if matched_plain and (payload.get('matched_protocol',{}).get('arm')!='S' or self.cfg.get('backbone_conditional_adapter',{}).get('enabled')):
            raise ValueError('Unsupported non-Stage2 matched overlay')
        self.engine=load_engine('stage2' if matched_plain else self.stage)
        self.cfg['small_llm']['model_name']=str(Path(model_dir).resolve());self.cfg['device']=device
        self.engine.set_seed(self.cfg['seed'])
        if self.device.type=='cuda':self.engine.set_cuda_seed(self.cfg['seed'])
        self.model,self.tokenizer,_=self.engine.build_small_llm_regressor(self.cfg);self.model.to(self.device)
        self.engine.freeze_small_llm_layers(self.model,24);self.auxiliary=None
        if self.stage=='stage2':
            if self.cfg.get('experiment',{}).get('variant','full')!='full':
                raise ValueError('Only complete Stage2 checkpoints are supported')
            if self.cfg.get('matched_protocol',{}).get('protocol_version')!='original_source_stage2_textfix_20261104':raise ValueError('Checkpoint is outside the Stage2 training protocol')
            self.engine.validate_text_only_cfg(self.cfg,required=True)
            if payload.get('soft_condition_label_head'):
                self.auxiliary=self.engine._setup_soft_condition_label_supervision(self.cfg,self.model,self.device)
            modules={'soft_condition_label_head':self.auxiliary} if self.auxiliary is not None else None
            self.engine._load_checkpoint(self.model,Path(checkpoint),cfg=self.cfg,load_head='always',matched_stage1=False,extra_modules=modules)
            if self.auxiliary is not None:self.auxiliary.eval()
        else:
            if matched_plain:
                self.engine._load_checkpoint(self.model,Path(checkpoint),cfg=self.cfg,load_head='always',matched_stage1=False)
            else:storage.restore(self.model,checkpoint,self.cfg)
        self.norm=payload['va_norm_stats'];self.model.eval();self.reporter=reporting_predict(self.engine,self.stage)
        del payload

    def predict(self,rows,*,development=False):
        if not rows:return []
        reporter=self.engine._predict if development else self.reporter
        predictions=reporter(self.model,input_rows(rows),self.tokenizer,self.device,self.cfg,va_norm_stats=self.norm)
        return associate(predictions,rows)
