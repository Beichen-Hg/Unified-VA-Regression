"""Frozen accepted profiles with explicit paths and two ordered 12/3 partitions."""
from pathlib import Path
import copy,math
from .io import read,write

PACKAGE=Path(__file__).resolve().parent
MODEL_ID='Qwen/Qwen3-8B-Base'
REVISION='49e3418fbbbca6ecbdf9608b4d22e5a407081db4'
ORIGINAL_TRAIN=('eng_restaurant','deu_politics','eng_laptop','zho_laptop','ukr_restaurant',
    'zho_restaurant','jpn_hotel','jpn_finance','rus_restaurant','zho_finance',
    'zho_environmental_protection','pcm_politics')
ORIGINAL_HELD=('eng_environmental_protection','tat_restaurant','swa_politics')
TARGETS=ORIGINAL_TRAIN+ORIGINAL_HELD
ALT_HELD=('rus_restaurant','ukr_restaurant','zho_finance')


def split_targets(split):
    if split not in ('original','alt1'):raise ValueError('Use original or alt1')
    held=ORIGINAL_HELD if split=='original' else ALT_HELD
    return tuple(t for t in TARGETS if t not in held),held


def target_config(target,data_root):
    if target not in TARGETS:raise ValueError('Unregistered target')
    language,domain=target.split('_',1)
    track='B' if domain in ('politics','environmental_protection') else 'A'
    suffix='alltasks' if domain in ('restaurant','laptop','hotel') else 'task1'
    return dict(task='small_llm_va',track=track,lang=language,domain=domain,
        data_dir=str(Path(data_root).resolve()/f'track_{track.lower()}'/'subtask_1'/language),
        train_file=f'{target}_train_{suffix}.jsonl',dev_file=f'{target}_dev_task1.jsonl',
        test_file=f'{target}_test_task1.jsonl',filter_unclear=False)


def template(stage,split):
    if stage not in ('baseline','stage1','stage2'):raise ValueError('Unknown stage')
    split_targets(split)
    return read(PACKAGE/'configs'/f'{stage}_{split}.json')


def substitute(value,replacements):
    if isinstance(value,dict):return {k:substitute(v,replacements) for k,v in value.items()}
    if isinstance(value,list):return [substitute(v,replacements) for v in value]
    if isinstance(value,str):
        for key,replacement in replacements.items():value=value.replace(key,str(replacement))
    return value


def prepare(stage,split,seed,data_root,model_dir,run_dir,device='cuda:0',parent=None):
    if isinstance(seed,bool) or not isinstance(seed,int):raise ValueError('seed must be an integer')
    if device!='cpu' and not device.startswith('cuda:'):raise ValueError('Specify cpu or cuda:N explicitly')
    run=Path(run_dir).resolve();data=Path(data_root).resolve();model=Path(model_dir).resolve()
    if (run/'STARTED.json').exists() or (run/'RESULT.json').exists():raise FileExistsError('Run already entered; choose a new run directory')
    run.mkdir(parents=True,exist_ok=True)
    train,held=split_targets(split)
    registry=run/'configs/eval';registry.mkdir(parents=True,exist_ok=True)
    for target in TARGETS:write(registry/f'{target}.yaml',target_config(target,data))
    # Keep the source's empty primary train input without modifying official data.
    primary=run/'configs/primary';primary.mkdir(parents=True,exist_ok=True)
    (primary/'empty_placeholder.jsonl').touch(exist_ok=True)
    cfg=substitute(template(stage,split),{'__DATA_ROOT__':data,'__MODEL_DIR__':model,'__RUN_DIR__':run,
        '__DEVICE__':device,'__PARENT_CHECKPOINT__':run/'checkpoints/shared_stage1.pt'})
    cfg['seed']=seed
    if 'sampler_seed' in cfg:cfg['sampler_seed']=seed
    cfg['data_dir']=str(primary)
    primary_target=target_config('jpn_hotel',data)
    for key in ('dev_file','test_file'):cfg[key]=str(Path(primary_target['data_dir'])/primary_target[key])
    cfg['macro_dev_eval']['targets']=[str(registry/f'{name}.yaml') for name in train]
    cfg['experiment'].update(variant='full',run_dir=str(run),data_root=str(data),model_dir=str(model),
        training_targets=list(train),held_out_targets=list(held),parent_run=str(Path(parent).resolve()) if parent else None)
    if stage=='stage2':
        cfg['matched_protocol'].update(author_approval='approved',execution_authorized=True)
    destination=run/'config.json'
    if destination.exists() and read(destination)!=cfg:raise RuntimeError('Existing plan has a different effective configuration')
    write(destination,cfg)
    return cfg


def expected_events(stage,split):
    cfg=template(stage,split)
    return cfg['experiment']['development_events']


def validate_effective(cfg):
    record=cfg['experiment'];original=template(record['stage'],record['split'])
    for key in ('lr','n_epochs','grad_accum','batch_size','grad_clip','scheduler','warmup_ratio','weight_decay',
                'loss','freeze_until','head_type','max_len','eval_every_updates','va_target_normalization','va_calibration'):
        if cfg[key]!=original[key]:raise ValueError('Frozen recipe changed: '+key)
    for key in ('pooling','prompt_template','feature_dropout','gradient_checkpointing','torch_dtype','attn_implementation'):
        if cfg['small_llm'][key]!=original['small_llm'][key]:raise ValueError('Frozen input/model rule changed: '+key)
    train,held=split_targets(record['split'])
    if [f"{s['lang']}_{s['domain']}" for s in cfg['auxiliary_data']['sources']]!=list(train):raise ValueError('Training partition/order changed')
    if [Path(p).stem for p in cfg['macro_dev_eval']['targets']]!=list(train):raise ValueError('Development partition/order changed')
    if cfg['run_test'] is not False or cfg.get('va_calibration'):raise ValueError('Test selection/calibration is not part of training')
    if record['stage']=='stage2':
        if cfg['backbone_conditional_adapter']!=original['backbone_conditional_adapter']:raise ValueError('Adapter settings changed')
        if cfg['soft_condition_label_supervision']!=original['soft_condition_label_supervision']:raise ValueError('Auxiliary supervision changed')
        if cfg['macro_dev_eval']['selection']!=original['macro_dev_eval']['selection']:raise ValueError('DP weights changed')
        if cfg.get('text_only_inference') is not True or cfg.get('disable_explicit_metadata') is not True:raise ValueError('Metadata-free inference flags changed')
    expected=3*math.ceil(record['train_aspects']/16) if record['stage']!='stage2' else math.ceil(record['train_aspects']/16)
    if record['expected_updates']!=expected:raise ValueError('Training budget changed')
    return True
