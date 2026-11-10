"""Full parent restoration and discarded one-backward interface/gradient checks."""
from contextlib import nullcontext
from pathlib import Path
import gc
from .io import read,write,now


def raw(model,tokenizer,rows,cfg,device):
    import torch
    values=[];model.eval()
    for row in rows:
        prompt=cfg['small_llm']['prompt_template'].format(text=row['text'],aspect=row['aspect'])
        tokens=tokenizer(prompt,return_tensors='pt',padding='max_length',truncation=True,max_length=cfg['max_len'])
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with torch.no_grad(),context:value=model(input_ids=tokens['input_ids'].to(device),attention_mask=tokens['attention_mask'].to(device))
        values.append(value.detach().float().cpu())
    return torch.cat(values)


def check_parent(engine,cfg):
    import numpy as np
    import torch
    device=torch.device(cfg['device']);run=Path(cfg['experiment']['run_dir']);parent_run=Path(cfg['experiment']['parent_run'])
    payload=torch.load(cfg['pretrain_from'],map_location='cpu',mmap=True,weights_only=False)
    pcfg=payload['effective_inference_config'];engine.set_seed(cfg['seed'])
    if device.type=='cuda':engine.set_cuda_seed(cfg['seed'])
    source,tokenizer,_=engine.build_small_llm_regressor(pcfg);source.to(device);engine.freeze_small_llm_layers(source,24)
    engine._load_checkpoint(source,Path(cfg['pretrain_from']),cfg=pcfg,load_head='always',matched_stage1=False)
    targets=engine._build_macro_dev_targets(pcfg);selected=read(parent_run/'RESULT.json')['selected_update']
    expected=read(parent_run/'development'/f'update{selected:06d}'/'METRICS.json');differences={};probes=[]
    for target in targets:
        pred=engine._predict(source,target['dev_data'],tokenizer,device,target['eval_cfg'],va_norm_stats=payload['va_norm_stats'])
        values=[np.asarray([p[key] for p in pred],dtype=float) for key in ['v_pred','a_pred','v_true','a_true']]
        score=float(engine.rmse_va(*values));differences[target['name']]=abs(score-expected['per_target'][target['name']])
        probes.extend(target['dev_data'][:2])
    if len(differences)!=12 or max(differences.values())>1e-6:raise RuntimeError('Selected Stage1 cannot be restored in the Stage2 model')
    probes=probes[:24]
    reference=raw(source,tokenizer,probes,cfg,device)
    del source,tokenizer;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    engine.set_seed(cfg['seed'])
    if device.type=='cuda':engine.set_cuda_seed(cfg['seed'])
    model,tokenizer,_=engine.build_small_llm_regressor(cfg);model.to(device);engine.freeze_small_llm_layers(model,24)
    auxiliary=engine._setup_soft_condition_label_supervision(cfg,model,device)
    engine._load_checkpoint(model,Path(cfg['pretrain_from']),cfg=cfg,load_head='always',matched_stage1=True,
        extra_modules={'soft_condition_label_head':auxiliary})
    initial=raw(model,tokenizer,probes,cfg,device)
    changed=[dict(row,lang='foreign',domain='foreign',id='foreign',v=-999,a=999) for row in probes]
    removed=[{'text':row['text'],'aspect':row['aspect']} for row in probes]
    zero_difference=float((reference-initial).abs().max())
    metadata_difference=[float((initial-raw(model,tokenizer,rows,cfg,device)).abs().max()) for rows in [changed,removed]]
    if zero_difference>1e-6 or metadata_difference!=[0.,0.]:raise RuntimeError('Zero-residual identity or metadata-free input probe failed')
    model.train();auxiliary.train()
    batch=next(iter(engine.build_dimasr_dataloader([probes[0]],tokenizer,batch_size=1,shuffle=False,max_len=cfg['max_len'],
        input_format=cfg['input_format'],prompt_template=cfg['small_llm']['prompt_template'],disable_explicit_metadata=True,num_workers=0)))
    context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context:
        prediction=model(input_ids=batch['input_ids'].to(device),attention_mask=batch['attention_mask'].to(device))
        target=engine._normalize_va(batch['va'].to(device),payload['va_norm_stats'])
        va_loss=engine.va_loss(prediction,target,mode=cfg['loss'])
        auxiliary_loss,stats=engine._compute_soft_condition_label_supervision_loss(auxiliary,model,batch,cfg,device)
        loss=va_loss+auxiliary_loss
    loss.backward()
    from .train import gradient_audit
    groups=gradient_audit(model)
    if groups.get('upper',{}).get('nonzero_gradient_tensors',0)==0 or groups.get('adapter',{}).get('nonzero_gradient_tensors',0)==0:
        raise RuntimeError('Missing Stage2 upper-block or adapter gradients')
    for adapter in model.backbone_conditional_adapters:
        for name in ('language','domain','composition','shared'):
            prefix={'language':'lang','domain':'domain','composition':'comp','shared':'shared'}[name]
            branch=getattr(adapter,prefix+'_up')
            has_gradient=all(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in branch.parameters())
            if not has_gradient:raise RuntimeError('A complete Stage2 branch lacks a scratch gradient path')
    write(run/'PREFLIGHT.json',dict(passed=True,completed_at=now(),full_development_per_target_abs_difference=differences,
        probes=len(probes),same_short_prompt_zero_adapter_raw_max_abs_difference=zero_difference,
        perturbed_raw_max_abs_difference=metadata_difference[0],removed_raw_max_abs_difference=metadata_difference[1],
        gradient_groups=groups,auxiliary_stats=stats,optimizer_updates=0,scratch_discarded=True,
        scope='Full registered parent development restoration and24 development input probes; no test data'))
    del model,tokenizer,auxiliary,payload,prediction,target,va_loss,auxiliary_loss,loss,batch,branch,adapter
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    # The unmodified training engine resets the seed and constructs a fresh model.
