"""Paired record-cluster statistics and development-only DP sensitivity."""
from pathlib import Path
from .io import read,write
from .config import TARGETS,split_targets


def evaluation_rows(folder):
    folder=Path(folder);report=read(folder/'RESULT.json')
    if not report.get('passed'):raise ValueError('Incomplete evaluation')
    rows=[]
    for target in TARGETS:
        rows.extend(dict(row,target=target) for row in read(folder/'predictions'/f'{target}.json'))
    return report,rows


def compare(left,right,split,output,resamples=10000):
    from .paired_statistics import bootstrap
    if len(left)!=len(right) or not left:raise ValueError('Provide equally many left/right seed evaluations')
    pairs=[];counts=None
    from collections import Counter
    for a,b in zip(left,right):
        ar,ap=evaluation_rows(a);br,bp=evaluation_rows(b)
        if ar['seed']!=br['seed'] or ar['split']!=split or br['split']!=split:raise ValueError('Pair seed/partition mismatch')
        pairs.append(dict(seed=ar['seed'],S=ap,T=bp))
        current=dict(Counter(p['target'] for p in ap))
        if counts is None:counts=current
        elif counts!=current:raise ValueError('Seed cohort counts differ')
    report=bootstrap(pairs,list(TARGETS),list(split_targets(split)[0]),counts,resamples=resamples,random_seed=20261101)
    report.update(difference='right minus left; negative favors right',fixed_model_intervals_not_training_seed_variance=True,
        seed_means_not_prediction_ensembles=True)
    write(output,report)
    return report


def dp_sensitivity(run_dir,output):
    from .source import load_engine
    engine=load_engine('stage2');run_dir=Path(run_dir);cfg=read(run_dir/'config.json')
    if cfg['experiment']['stage']!='stage2':raise ValueError('DP analysis requires a Stage2 archive')
    original=cfg['macro_dev_eval']['selection'];events=[read(p) for p in (run_dir/'outputs/evaluation_archive').glob('*.json')]
    if len(events)!=len(cfg['experiment']['development_events']):raise ValueError('Incomplete DP development archive')
    import copy
    rules=[('default',copy.deepcopy(original))]
    groups={'tail':['worst_k_weight'],'pseudo_ood':['pseudo_ood_weight','pseudo_ood_worst_k_weight'],
        'spread':['std_gap_v_weight','std_gap_a_weight'],'floor':['ccc_macro_floor_weight','pcc_macro_floor_weight']}
    for group,fields in groups.items():
        for multiplier in [0.,.5,2.]:
            variant=copy.deepcopy(original)
            for field in fields:variant[field]=original[field]*multiplier
            rules.append((group+'_'+str(multiplier),variant))
    rules.append(('macro_RMSE_only',{'mode':'macro_mean'}));results=[]
    for name,selection in rules:
        values=[dict(update=e['global_update'],score=float(engine.compute_macro_dev_selection(e['per_target'],selection)[0])) for e in events]
        results.append(dict(rule=name,selected=min(values,key=lambda r:(r['score'],r['update'])),all_development_scores=values))
    write(output,dict(development_only=True,formal_checkpoint_not_changed=True,test_not_used=True,rules=results))
    return results
