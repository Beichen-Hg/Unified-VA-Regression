"""Portable lossless overlays and an identical-tensor Stage1-to-Stage2 bridge."""
from pathlib import Path
import copy,os,shutil
from .io import read,write,now
from .config import MODEL_ID,REVISION


def reference(cfg):
    return dict(repository=MODEL_ID,revision=REVISION,tokenizer_revision=REVISION,
        torch_dtype=cfg['small_llm']['torch_dtype'],freeze_until=cfg['freeze_until'])


def cpu_state(module):
    return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def overlay(model,cfg,epoch,update,metrics,norm):
    parameters=dict(model.backbone.named_parameters());state=model.backbone.state_dict()
    kept={k for k in state if k not in parameters or parameters[k].requires_grad}
    nonpersistent={k:v.detach().cpu().clone() for k,v in model.backbone.named_buffers(remove_duplicate=False) if k not in state}
    return dict(checkpoint_type='unified_va_stage1_overlay',epoch=epoch,global_update=update,metrics=metrics,
        backbone={k:state[k].detach().cpu().clone() for k in sorted(kept)},head=cpu_state(model.head),
        pool_gate=cpu_state(model.pool_gate) if model.pool_gate is not None else None,
        head_type=cfg['head_type'],pooling=cfg['small_llm']['pooling'],freeze_until=cfg['freeze_until'],
        effective_inference_config=copy.deepcopy(cfg),va_norm_stats=norm,
        checkpoint_storage=dict(format='unified_va_stage1_overlay_v1',base_reference=reference(cfg),
            saved_backbone_names=sorted(kept),omitted_frozen_base_names=sorted(set(state)-kept),
            backbone_schema={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in state.items()},
            nonpersistent_buffers=nonpersistent,lossless=True))


def save(path,payload,min_free_gib=40):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    # The size check covers the complete new tensor payload before the write.
    tensors=[]
    def collect(value):
        if torch.is_tensor(value):tensors.append(value)
        elif isinstance(value,dict):
            for v in value.values():collect(v)
    collect(payload)
    estimated=sum(v.numel()*v.element_size() for v in tensors)+32*1024**2
    if shutil.disk_usage(path.parent).free-estimated<min_free_gib*2**30:raise RuntimeError('Insufficient checkpoint storage reserve')
    temporary=path.with_name(path.name+'.tmp')
    if temporary.exists():raise FileExistsError('Partial checkpoint exists')
    torch.save(payload,temporary);os.replace(temporary,path)


def link(source,target):
    source,target=Path(source).resolve(),Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() and source==target.resolve():return
    temporary=target.with_name(target.name+'.linktmp')
    if temporary.exists():raise FileExistsError(str(temporary))
    try:os.link(source,temporary)
    except OSError:
        # Cross-filesystem fallback; no numerical re-encoding.
        shutil.copyfile(source,temporary)
    os.replace(temporary,target)


def restore(model,path,cfg):
    import torch
    payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    storage=payload['checkpoint_storage']
    if storage.get('format') not in ('unified_va_stage1_overlay_v1','absolute_tensor_overlay_v1'):raise ValueError('Unsupported Stage1 overlay')
    if 'base_reference' in storage and storage['base_reference']!=reference(cfg):raise ValueError('Different pinned base')
    incoming=payload['backbone'];current=model.backbone.state_dict()
    if set(current)-set(incoming)!=set(storage['omitted_frozen_base_names']) or set(incoming)-set(current):raise ValueError('Stage1 overlay coverage differs')
    if any(current[k].shape!=v.shape or current[k].dtype!=v.dtype for k,v in incoming.items()):raise ValueError('Stage1 tensor schema differs')
    model.backbone.load_state_dict(incoming,strict=False);model.head.load_state_dict(payload['head'],strict=True)
    if model.pool_gate is not None:model.pool_gate.load_state_dict(payload['pool_gate'],strict=True)
    elif payload.get('pool_gate') is not None:raise ValueError('Wrong pooling checkpoint')
    target_buffers=dict(model.backbone.named_buffers(remove_duplicate=False))
    for key,value in storage.get('nonpersistent_buffers',{}).items():target_buffers[key].copy_(value)
    for module,state in [(model.backbone,incoming),(model.head,payload['head']),(model.pool_gate,payload.get('pool_gate'))]:
        if module is not None and any(not torch.equal(module.state_dict()[k].detach().cpu(),v) for k,v in state.items()):raise ValueError('Lossless restored tensor differs')
    return payload


def bridge(parent_run,cfg):
    import torch
    parent=Path(parent_run).resolve();result=read(parent/'RESULT.json')
    if result.get('passed') is not True or result.get('stage')!='stage1':raise ValueError('Stage2 needs a completed selected Stage1 run')
    source=Path(result['checkpoint'])
    if not source.is_absolute():source=parent/source
    payload=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    pcfg=copy.deepcopy(payload['effective_inference_config'])
    if pcfg['experiment']['split']!=cfg['experiment']['split'] or pcfg['seed']!=cfg['seed']:raise ValueError('Stage1 seed/partition differs')
    if result['updates']!=pcfg['experiment']['expected_updates'] or result['development_checks']!=len(pcfg['experiment']['development_events']):raise ValueError('Incomplete Stage1 budget')
    if result['selected_update']!=payload['global_update'] or result['selected_update'] not in pcfg['experiment']['development_events']:
        raise ValueError('Stage1 selected checkpoint/update mismatch')
    if read(parent/'FINAL_DEV_RESTORE.json').get('passed') is not True:raise ValueError('Selected Stage1 development restoration has not passed')
    storage=payload['checkpoint_storage']
    if storage.get('format')!='unified_va_stage1_overlay_v1':raise ValueError('The portable bridge needs a Stage1 run produced by this package')
    pcfg['small_llm']['model_name']=cfg['small_llm']['model_name']
    pcfg['checkpoint_storage']={'format':'pinned_base_overlay_v1'}
    pcfg['matched_protocol']=copy.deepcopy(cfg['matched_protocol'])
    pcfg['matched_protocol'].update(arm='S',protocol_version='recovered_original_stage1_source_20261104',
        expected_total_updates=pcfg['experiment']['expected_updates'],selection_policy='original_macro_mean_on_two_decimal_predictions')
    pcfg.update(sampler_seed=cfg['seed'],disable_explicit_metadata=True,text_only_inference=False,
        legacy_prediction_rounding='python_two_decimals_before_metrics')
    pcfg['macro_dev_eval']['targets']=cfg['macro_dev_eval']['targets']
    converted=dict(payload)
    converted.update(checkpoint_type='pinned_base_overlay',matched_protocol=pcfg['matched_protocol'],effective_inference_config=pcfg,
        checkpoint_storage=dict(format='pinned_base_overlay_v1',base_reference=reference(cfg),
            backbone_schema=storage['backbone_schema'],omitted_frozen_base_keys=storage['omitted_frozen_base_names'],
            nonpersistent_buffers=storage['nonpersistent_buffers'],tensor_encoding='absolute unchanged tensors; no optimization or quantization'))
    destination=Path(cfg['pretrain_from'])
    if destination.exists():raise FileExistsError('Stage1 bridge already exists; choose a fresh Stage2 run')
    save(destination,converted)
    reloaded=torch.load(destination,map_location='cpu',mmap=True,weights_only=False)
    for field in ('backbone','head','pool_gate'):
        original=payload[field] or {};new=reloaded[field] or {}
        if set(original)!=set(new) or any(not torch.equal(original[k],new[k]) for k in original):raise RuntimeError('Bridge changed an S tensor')
    write(Path(cfg['experiment']['run_dir'])/'PARENT_BRIDGE.json',dict(passed=True,created_at=now(),
        parent_run=str(parent),selected_update=result['selected_update'],same_all_saved_tensors=True,optimizer_updates=0))
    return parent
