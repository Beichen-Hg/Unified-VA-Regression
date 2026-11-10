"""Small public command surface; every source engine runs in a fresh process."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from .config import prepare
from .io import read


def worker(arguments,cpu=False):
    env=dict(os.environ,USE_TF='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    if cpu:env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run([sys.executable,'-m','unified_va._worker',*arguments],env=env,check=True)


def add_run_arguments(parser,stage=None):
    if stage is None:parser.add_argument('--stage',choices=['baseline','stage1','stage2'],required=True)
    parser.add_argument('--split',choices=['original','alt1'],required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--data-root',required=True,help='Official task-dataset directory')
    parser.add_argument('--model-dir',required=True,help='Local pinned Qwen3-8B-Base snapshot')
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--device',default='cuda:0',help='Explicit cpu or cuda:N; no automatic GPU selection')
    parser.add_argument('--parent',help='Completed Stage1 run directory, for Stage2')


def parser():
    result=argparse.ArgumentParser(prog='experiment',description='Unified multilingual and multidomain valence-arousal regression')
    commands=result.add_subparsers(dest='command',required=True)
    add_run_arguments(commands.add_parser('plan',help='Create a frozen run plan and audit train/dev data; no model loading'))
    for stage in ['baseline','stage1','stage2']:
        p=commands.add_parser('train-'+stage,help='Train the '+stage+' recipe in a fresh output directory')
        add_run_arguments(p,stage);p.set_defaults(stage=stage)
    p=commands.add_parser('evaluate',help='Full dev restoration, then the fixed15-target final test')
    p.add_argument('--run-dir',required=True);p.add_argument('--data-root',required=True);p.add_argument('--model-dir',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--output')
    for name in ['predict','diagnose']:
        p=commands.add_parser(name,help='Text/aspect prediction' if name=='predict' else 'Original auxiliary-head condition decodability')
        p.add_argument('--checkpoint',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--input',required=True)
        p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p=commands.add_parser('compare',help='Seed-paired macro RMSE and record-cluster bootstrap')
    p.add_argument('--left',nargs='+',required=True);p.add_argument('--right',nargs='+',required=True)
    p.add_argument('--split',choices=['original','alt1'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--resamples',type=int,default=10000)
    p=commands.add_parser('dp-sensitivity',help='Offline local reweighting of saved dev scores; does not change selected models')
    p.add_argument('--run-dir',required=True);p.add_argument('--output',required=True)
    commands.add_parser('show-results',help='Print the included validated aggregate results')
    return result


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command=='plan' or args.command.startswith('train-'):
        if args.stage=='stage2' and not args.parent:raise SystemExit('Stage2 requires --parent pointing to a completed Stage1 run')
        cfg=prepare(args.stage,args.split,args.seed,args.data_root,args.model_dir,args.run_dir,args.device,args.parent)
        path=str(Path(args.run_dir).resolve()/'config.json')
        worker(['audit',path],cpu=True)
        if args.command!='plan':worker(['train',path])
        print(json.dumps(dict(run_dir=cfg['experiment']['run_dir'],stage=args.stage,split=args.split,seed=args.seed),indent=2))
    elif args.command=='show-results':
        data=read(Path(__file__).resolve().parent/'resources/mainline_summary.json')
        for split in ['original','alt1']:
            full=next(r for r in data[split]['group_seed_statistics'] if r['group']=='full')
            print(f"{split}: Stage1 {full['S_mean']:.6f} +/- {full['S_sample_std']:.6f}; Stage2 {full['T_mean']:.6f} +/- {full['T_sample_std']:.6f}")
    else:
        worker(['command',json.dumps(vars(args))],cpu=args.command in ['compare','dp-sensitivity'])
