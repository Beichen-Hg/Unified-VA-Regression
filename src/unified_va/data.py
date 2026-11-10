"""Official-format training audit; test/held-out train/dev files are not used here."""
from collections import Counter
from pathlib import Path
import os
import numpy as np
from .config import split_targets,target_config,validate_effective
from .io import write,now
from .source import load_engine


def audit(cfg):
    if os.environ.get('DIMABSA2026_DATA_DIR'):raise ValueError('Unset DIMABSA2026_DATA_DIR; use the explicit --data-root')
    validate_effective(cfg);stage=cfg['experiment']['stage'];engine=load_engine(stage)
    train,held=split_targets(cfg['experiment']['split']);rows=[];counts=[]
    for name in train:
        current=target_config(name,cfg['experiment']['data_root'])
        item=dict(target=name)
        for split in ('train','dev'):
            path=Path(current['data_dir'])/current[split+'_file']
            if not path.is_file():raise FileNotFoundError(str(path))
            expanded=engine.expand_to_aspect_level(engine.load_samples(current,split))
            item[split+'_aspects']=len(expanded)
            if split=='train':rows.extend(expanded)
        counts.append(item)
    norm=engine._build_va_norm_stats(rows,True)
    if len(rows)!=cfg['experiment']['train_aspects'] or sum(i['dev_aspects'] for i in counts)!=cfg['experiment']['dev_aspects']:
        raise ValueError('Dataset counts differ from the accepted dataset revision')
    result=dict(passed=True,checked_at=now(),stage=stage,split=cfg['experiment']['split'],
        training_aspects=len(rows),development_aspects=sum(i['dev_aspects'] for i in counts),
        ordered_training_targets=list(train),held_out_targets=list(held),normalization=norm,
        langs=sorted({r['lang'] for r in rows}),domains=sorted({r['domain'] for r in rows}),
        target_counts=counts,test_files_read=0,held_out_train_dev_files_read=0)
    if stage=='stage2':
        expected=cfg['matched_protocol']['train_only_normalization']
        if norm!=expected:raise ValueError('Recomputed train normalization differs from the fixed data revision')
        adapter=cfg['backbone_conditional_adapter']
        if adapter['langs']!=result['langs'] or adapter['domains']!=result['domains']:raise ValueError('Condition vocabulary differs from TRAIN records')
    write(Path(cfg['experiment']['run_dir'])/'DATA_AUDIT.json',result)
    return result
