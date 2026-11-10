"""Audit/storage adapters around the stage-specific training engines."""
from pathlib import Path
import gc,inspect,time,shutil
from .config import validate_effective
from .io import read,write,now
from .source import load_engine
from . import storage


def rmse_map(values):
    return {name:float(value['rmse_va'] if isinstance(value,dict) else value) for name,value in values.items()}


def gradient_audit(model):
    import torch
    groups={}
    for name,parameter in model.named_parameters():
        group=('adapter' if '.adapter.' in name else 'upper' if name.startswith('backbone.layers.') else
               'norm' if name.startswith('backbone.norm.') else 'readout' if name.startswith(('head.','pool_gate.')) else 'other')
        record=groups.setdefault(group,dict(requires_grad_tensors=0,gradient_tensors=0,nonzero_gradient_tensors=0))
        record['requires_grad_tensors']+=int(parameter.requires_grad)
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():raise RuntimeError('Nonfinite gradient before optimizer step')
            record['gradient_tensors']+=1
            record['nonzero_gradient_tensors']+=int(torch.count_nonzero(parameter.grad).item()>0)
    return groups


class Hooks:
    def __init__(self,engine,cfg):
        self.engine,self.cfg=engine,cfg
        self.run=Path(cfg['experiment']['run_dir']);self.output=self.run/'outputs'
        self.output.mkdir(parents=True,exist_ok=False)
        self.role=cfg['experiment']['stage'];self.events=[];self.capture=None;self.model=None;self.norm=None
        self.steps=0;self.final_reload=False;self.restored_update=None;self.patches=[]
        self.started=time.monotonic();self.original_step=None

    def patch(self,name,value):
        self.patches.append((name,getattr(self.engine,name)));setattr(self.engine,name,value)

    def install(self):
        import torch
        engine=self.engine
        original_build=engine.build_small_llm_regressor
        original_predict=engine._predict;original_macro=engine._evaluate_macro_dev_rmse
        original_norm=engine._build_va_norm_stats;original_load=engine._load_checkpoint
        original_save=engine._save_checkpoint
        def build(cfg):
            result=original_build(cfg);self.model=result[0]
            return result
        def predict(*args,**kwargs):
            result=original_predict(*args,**kwargs)
            if self.capture is not None:self.capture.append(result)
            return result
        def normalization(data,enabled):
            self.norm=original_norm(data,enabled)
            expected=read(self.run/'DATA_AUDIT.json')['normalization']
            if self.norm!=expected:raise RuntimeError('Formal training normalization differs from audited train rows')
            return self.norm
        def macro(model,targets,tokenizer,device,**kwargs):
            self.capture=[]
            try:result=original_macro(model,targets,tokenizer,device,**kwargs);predictions=self.capture
            finally:self.capture=None
            value,details=result;per_target=rmse_map(details)
            if len(per_target)!=12 or len(predictions)!=12 or sum(map(len,predictions))!=self.cfg['experiment']['dev_aspects']:
                raise RuntimeError('Incomplete registered development evaluation')
            if self.final_reload:
                expected=next(event for event in self.events if event['update']==self.restored_update)
                differences={k:abs(per_target[k]-expected['per_target'][k]) for k in per_target}
                passed=abs(float(value)-expected['macro_dev_rmse_va'])<=1e-6 and max(differences.values())<=1e-6
                write(self.run/'FINAL_DEV_RESTORE.json',dict(passed=passed,selected_update=self.restored_update,
                    macro_abs_difference=abs(float(value)-expected['macro_dev_rmse_va']),per_target_abs_difference=differences))
                if not passed:raise RuntimeError('Full selected-model development restore failed')
                return result
            frame=inspect.currentframe().f_back
            epoch,update=int(frame.f_locals['epoch']),int(frame.f_locals['global_update']);del frame
            if update not in self.cfg['experiment']['development_events'] or any(e['update']==update for e in self.events):
                raise RuntimeError('Unexpected/duplicate development event')
            names=[target['name'] for target in targets if target['dev_data']]
            folder=self.run/'development'/f'update{update:06d}'
            for name,rows in zip(names,predictions):write(folder/f'{name}.json',rows)
            event=dict(update=update,epoch=epoch,macro_dev_rmse_va=float(value),per_target=per_target)
            write(folder/'METRICS.json',event);self.events.append(event);write(self.run/'DEVELOPMENT_EVENTS.json',self.events)
            if self.role!='stage2':
                storage.save(self.run/'checkpoints'/f'update{update:06d}.pt',
                    storage.overlay(model,self.cfg,epoch,update,dict(macro_dev_rmse_va=float(value),per_target=per_target),self.norm))
            return result
        def save(model,run_dir,epoch,metrics,config,filename='checkpoint.pt',global_update=None,extra_modules=None):
            source=self.run/'checkpoints'/f'update{global_update:06d}.pt'
            if not source.is_file():raise RuntimeError('Checkpoint lacks a complete registered development event')
            storage.link(source,Path(run_dir)/filename)
        def load(model,path,cfg=None,load_backbone=True,load_head='if_compatible',**kwargs):
            if self.role=='stage2':
                payload,report=original_load(model,path,cfg=cfg,load_backbone=load_backbone,load_head=load_head,**kwargs)
            else:
                payload=storage.restore(model,path,self.cfg);report={'exact_saved_tensor_restore':True}
            if Path(path).resolve()==(self.output/'checkpoint.pt').resolve():
                self.final_reload=True;self.restored_update=payload['global_update']
            return payload,report
        def guarded_save(model,run_dir,*args,**kwargs):
            # Reserve the new overlay plus heads/auxiliary tensors before the write.
            params=dict(model.backbone.named_parameters())
            size=sum(v.numel()*v.element_size() for k,v in model.backbone.state_dict().items()
                     if k not in params or params[k].requires_grad)
            size+=sum(p.numel()*p.element_size() for p in model.head.parameters())+64*1024**2
            if shutil.disk_usage(self.run).free-size<40*2**30:raise RuntimeError('Stage2 checkpoint would breach the40GiB reserve')
            return original_save(model,run_dir,*args,**kwargs)
        self.patch('build_small_llm_regressor',build);self.patch('make_run_dir',lambda **kwargs:self.output)
        self.patch('_predict',predict);self.patch('_evaluate_macro_dev_rmse',macro)
        self.patch('_build_va_norm_stats',normalization);self.patch('_load_checkpoint',load)
        self.patch('_save_checkpoint',guarded_save if self.role=='stage2' else save)
        self.original_step=torch.optim.AdamW.step
        def step(optimizer,*args,**kwargs):
            if self.steps==0:
                groups=gradient_audit(self.model);write(self.run/'FIRST_OPTIMIZER_GRADIENTS.json',groups)
                if self.role!='stage2' and groups.get('upper',{}).get('gradient_tensors',0)!=0:
                    raise RuntimeError('Source Stage1 checkpointing behavior changed; do not silently alter the recipe')
                if self.role=='stage2' and len(self.model.backbone_conditional_adapters)!=12:
                    raise RuntimeError('Expected complete Stage2 adapters in layers24 through35')
            answer=self.original_step(optimizer,*args,**kwargs);self.steps+=1
            if self.steps==1 or self.steps%10==0 or self.steps==self.cfg['experiment']['expected_updates']:
                write(self.run/'PROGRESS.json',dict(checked_at=now(),completed_optimizer_updates=self.steps,
                    registered_updates=self.cfg['experiment']['expected_updates'],elapsed_seconds=time.monotonic()-self.started))
            return answer
        torch.optim.AdamW.step=step

    def close(self):
        import torch
        if self.original_step is not None:torch.optim.AdamW.step=self.original_step
        for name,original in reversed(self.patches):setattr(self.engine,name,original)
        self.model=None

    def finish(self,metrics):
        from .source import load_engine
        if self.steps!=self.cfg['experiment']['expected_updates'] or metrics['num_optimizer_updates']!=self.steps:
            raise RuntimeError('Incomplete optimizer budget')
        expected=self.cfg['experiment']['development_events']
        if [e['update'] for e in self.events]!=expected:raise RuntimeError('Incomplete development schedule')
        if self.role=='stage2':
            selector=self.engine.compute_macro_dev_selection
            events=[]
            for path in (self.output/'evaluation_archive').glob('*.json'):
                event=read(path)
                if event['selection_cfg']!=self.cfg['macro_dev_eval']['selection']:raise RuntimeError('DP drift')
                score,_=selector(event['per_target'],event['selection_cfg'])
                events.append(dict(update=event['global_update'],score=float(score),checkpoint=str(path.parent/event['checkpoint'])))
            if sorted(e['update'] for e in events)!=expected:raise RuntimeError('Incomplete recoverable DP archive')
            selected=min(events,key=lambda e:(e['score'],e['update']))
        else:
            events=[dict(e,checkpoint=str(self.run/'checkpoints'/f"update{e['update']:06d}.pt")) for e in self.events]
            selected=min(events,key=lambda e:e['macro_dev_rmse_va'])
        if selected['update']!=metrics['best_update'] or read(self.run/'FINAL_DEV_RESTORE.json')['passed'] is not True:
            raise RuntimeError('Source selection/restoration differs from the registered rule')
        final=self.run/'checkpoints'/(self.role+'.pt');storage.link(selected['checkpoint'],final)
        # The source has already restored this same selected model for full dev evaluation.
        storage.link(selected['checkpoint'],self.output/'checkpoint.pt')
        protected={selected['update'],max(expected)};retired=[]
        for event in events:
            path=Path(event['checkpoint']).resolve()
            if not path.is_relative_to(self.run):raise RuntimeError('Checkpoint outside the new run')
            if event['update'] not in protected:
                if path.is_symlink():raise RuntimeError('Unexpected external checkpoint link')
                retired.append(dict(update=event['update'],relative_path=str(path.relative_to(self.run)),bytes=path.stat().st_size))
        write(self.run/'RETENTION_PLAN.json',dict(protected_updates=sorted(protected),retire=retired))
        for event in events:
            if event['update'] not in protected:Path(event['checkpoint']).unlink()
        write(self.run/'RETENTION.json',dict(passed=True,retired=retired,all_development_predictions_and_scalars_retained=True))
        result=dict(passed=True,stage=self.role,split=self.cfg['experiment']['split'],seed=self.cfg['seed'],
            variant=self.cfg['experiment']['variant'],updates=self.steps,development_checks=len(events),
            selected_update=selected['update'],checkpoint=str(final.relative_to(self.run)),
            selection='DP' if self.role=='stage2' else '12-target macro development RMSE',test_used_for_selection=False)
        write(self.run/'RESULT.json',result)
        return result


def run(config_path):
    import torch
    cfg=read(config_path);validate_effective(cfg);run_dir=Path(cfg['experiment']['run_dir'])
    if (run_dir/'STARTED.json').exists():raise FileExistsError('Run has already started; automatic retry/resume is disabled')
    if not read(run_dir/'DATA_AUDIT.json')['passed']:raise RuntimeError('Training data audit is missing')
    if cfg['device'].startswith('cuda:') and not torch.cuda.is_available():raise RuntimeError('Requested CUDA device is unavailable')
    if not Path(cfg['small_llm']['model_name']).is_dir():raise FileNotFoundError('Local pinned base model directory is missing')
    write(run_dir/'STARTED.json',dict(started_at=now(),stage=cfg['experiment']['stage']))
    engine=load_engine(cfg['experiment']['stage']);torch.set_num_threads(4)
    if cfg['experiment']['stage']=='stage2':
        if not cfg['experiment']['parent_run']:raise ValueError('Stage2 needs --parent pointing to a completed Stage1 run')
        storage.bridge(cfg['experiment']['parent_run'],cfg)
        from .preflight import check_parent
        check_parent(engine,cfg)
    hooks=Hooks(engine,cfg);hooks.install()
    try:
        metrics=engine.train_small_llm_va(cfg)
        write(run_dir/'ENGINE_COMPLETE.json',metrics)
        return hooks.finish(metrics)
    finally:
        hooks.close();gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
