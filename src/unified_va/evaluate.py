"""Strict development restoration before the fixed 15-target final evaluation."""
from pathlib import Path
import copy,statistics
from .config import TARGETS,target_config
from .io import read,write,now
from .inference import Predictor


def rmse(rows):
    import math
    return math.sqrt(math.fsum((p['v_pred']-p['v_true'])**2+(p['a_pred']-p['a_true'])**2 for p in rows)/len(rows))


def run(run_dir,data_root,model_dir,device,output=None):
    run_dir=Path(run_dir).resolve();result=read(run_dir/'RESULT.json')
    if result.get('passed') is not True:raise ValueError('Training run has not completed')
    checkpoint=run_dir/result['checkpoint'];destination=Path(output).resolve() if output else run_dir/'evaluation'
    if (destination/'RESULT.json').exists():raise FileExistsError('Evaluation already complete; select a new output directory to repeat it')
    predictor=Predictor(checkpoint,model_dir,device)
    selected=read(run_dir/'development'/f"update{result['selected_update']:06d}"/'METRICS.json')
    differences={}
    for name in predictor.cfg['experiment']['training_targets']:
        ecfg=target_config(name,data_root)
        rows=predictor.engine.expand_to_aspect_level(predictor.engine.load_samples(ecfg,'dev'))
        predictions=predictor.predict(rows,development=True)
        differences[name]=abs(rmse(predictions)-selected['per_target'][name])
    if len(differences)!=12 or max(differences.values())>1e-6:raise RuntimeError('Selected checkpoint failed full development restoration')
    write(destination/'DEV_RESTORE.json',dict(passed=True,per_target_abs_difference=differences))
    scores={};count=0
    for name in TARGETS:
        ecfg=target_config(name,data_root)
        rows=predictor.engine.expand_to_aspect_level(predictor.engine.load_samples(ecfg,'test'))
        predictions=predictor.predict(rows)
        if not predictions:raise ValueError('Empty test target '+name)
        write(destination/'predictions'/f'{name}.json',predictions)
        scores[name]=rmse(predictions);count+=len(predictions)
        write(destination/'PROGRESS.json',dict(checked_at=now(),completed_targets=list(scores),aspects=count))
    if count!=19977:raise ValueError('The evaluation cohort contains 19977 aspects')
    final=dict(passed=True,stage=result['stage'],seed=result['seed'],split=result['split'],variant=result['variant'],
        checkpoint=str(checkpoint),aspects=count,Full=statistics.mean(scores.values()),per_target_rmse_va=scores,
        precision='native Torch float32 two-decimal prediction values',test_used_for_selection=False)
    write(destination/'RESULT.json',final)
    return final
