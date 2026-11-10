"""Select one private compatibility engine in each process."""
from pathlib import Path
import importlib,os,sys

ROLES=('baseline','stage1','stage2')


def load_engine(role):
    if role not in ROLES:raise ValueError('Unknown Unified VA regression stage')
    os.environ.update(USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    root=Path(__file__).resolve().parent/'engines'/role
    existing=sys.modules.get('src')
    if existing is not None and Path(existing.__file__).resolve().parent!=root/'src':
        raise RuntimeError('Run each compatibility engine in a separate process; use the Unified VA regression CLI')
    sys.path.insert(0,str(root))
    engine=importlib.import_module('src.train.train_small_llm_va')
    if Path(engine.__file__).resolve()!=root/'src/train/train_small_llm_va.py':
        raise RuntimeError('A different src package shadows the Unified VA regression engine')
    return engine
