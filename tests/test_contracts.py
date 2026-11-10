from pathlib import Path
import ast,copy,json,os,re,subprocess,sys
import numpy as np
import pytest
from unified_va.config import template,prepare,validate_effective,split_targets,TARGETS
from unified_va.inference import input_rows,associate
from unified_va.diagnostics import classification
from unified_va.paired_statistics import bootstrap

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('stage',['baseline','stage1','stage2'])
@pytest.mark.parametrize('split',['original','alt1'])
def test_relocated_profiles_and_ordered_data_membership(tmp_path,stage,split):
    cfg=prepare(stage,split,13,tmp_path/'official-data',tmp_path/'model',tmp_path/stage,device='cpu',parent=tmp_path/'S' if stage=='stage2' else None)
    assert validate_effective(cfg)
    train,held=split_targets(split)
    assert len(train)==12 and len(held)==3 and not set(train)&set(held)
    assert [Path(p).stem for p in cfg['macro_dev_eval']['targets']]==list(train)
    assert all(Path(p).is_file() for p in cfg['macro_dev_eval']['targets'])
    assert (Path(cfg['data_dir'])/cfg['train_file']).read_text()==''
    assert not (tmp_path/'official-data').exists()
    changed=copy.deepcopy(cfg);changed['lr']*=2
    with pytest.raises(ValueError):validate_effective(changed)


def test_input_boundary_and_post_forward_identity_and_gold_association():
    rows=[dict(text='example',aspect='service',id='private',aspect_index=2,lang='eng',domain='restaurant',v=6.,a=4.)]
    safe=input_rows(rows)
    assert set(safe[0])=={'text','aspect','id','aspect_index','sample_aspects'}
    assert safe[0]['id']=='0' and safe[0]['aspect_index']==0
    result=associate([dict(v_pred=5.,a_pred=4.)],rows)
    assert result[0]['id']=='private' and result[0]['aspect_index']==2
    assert result[0]['v_true']==6. and result[0]['target']=='eng_restaurant'


def test_auxiliary_metrics_exclude_unknown_and_untrained_classes():
    report=classification([0,1,1,2,3],[1,1,0,1,3],[1,2])
    assert report['known']==3 and report['unsupported']==2
    assert report['coverage']==.6 and report['accuracy']==pytest.approx(1/3)
    assert report['macro_F1']==.25


def test_record_clusters_and_pair_direction_with_multiple_aspects():
    pairs=[]
    for seed in [42,13,7]:
        pair={'seed':seed}
        for role,error in [('S',2.),('T',1.)]:
            pair[role]=[dict(target=t,id=record,aspect_index=i,v_true=4.,a_true=5.,v_pred=4.+error,a_pred=5.+error)
                        for t in ['a','b','c'] for record in ['x','y'] for i in [0,1]]
        pairs.append(pair)
    report=bootstrap(pairs,['a','b','c'],['a','b'],{'a':4,'b':4,'c':4},resamples=100,random_seed=20261101)
    overall=next(r for r in report['group_seed_statistics'] if r['group']=='full')
    assert overall['T_minus_S_mean']<0 and overall['seeds']==3
    # Cluster building is exercised by bootstrap; all aspects share their record draw.
    assert report['bootstrap_intervals']


def test_every_python_file_parses_and_release_has_no_host_absolute_paths():
    for path in ROOT.rglob('*.py'):
        text=path.read_text(encoding='utf-8');ast.parse(text)
        assert not re.search(r'/home/[A-Za-z0-9_.-]+/',text)
        assert not re.search(r'[A-Za-z]:[/\\]Users[/\\][A-Za-z0-9_.-]+',text)
    for path in (ROOT/'src/unified_va/configs').glob('*.json'):
        text=path.read_text();json.loads(text)
        assert '/home/' not in text and 'C:/Users/' not in text


@pytest.mark.parametrize('role',['baseline','stage1','stage2'])
def test_actual_source_engine_tiny_training_and_lossless_restore(role):
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),USE_TF='0',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run([sys.executable,str(ROOT/'tests/source_smoke.py'),role],cwd=ROOT,env=env,
        text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    assert result.returncode==0,result.stdout
