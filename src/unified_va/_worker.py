"""Process-isolated implementation behind the public CLI."""
import json,sys,traceback
from pathlib import Path
from .io import read,read_rows,write,write_rows


def main():
    action,value=sys.argv[1:3]
    if action=='audit':
        from .data import audit
        result=audit(read(value));print(json.dumps({k:result[k] for k in ['passed','training_aspects','development_aspects']},indent=2))
        return
    if action=='train':
        from .train import run
        cfg=read(value)
        try:result=run(value)
        except BaseException:
            write(Path(cfg['experiment']['run_dir'])/'FAILURE.json',dict(error=traceback.format_exc()));raise
        print(json.dumps(result,indent=2));return
    args=json.loads(value);command=args['command']
    if command=='evaluate':
        from .evaluate import run
        result=run(args['run_dir'],args['data_root'],args['model_dir'],args['device'],args['output'])
    elif command in ['predict','diagnose']:
        from .inference import Predictor
        predictor=Predictor(args['checkpoint'],args['model_dir'],args['device']);rows=read_rows(args['input'])
        if command=='predict':
            result=predictor.predict(rows);write_rows(args['output'],result)
        else:
            from .diagnostics import run
            result=run(predictor,rows);write(args['output'],result)
        print(json.dumps(dict(command=command,rows=len(rows),output=args['output'])));return
    elif command=='compare':
        from .analysis import compare
        result=compare(args['left'],args['right'],args['split'],args['output'],args['resamples'])
    elif command=='dp-sensitivity':
        from .analysis import dp_sensitivity
        result=dp_sensitivity(args['run_dir'],args['output'])
    else:raise ValueError('Unknown command')
    print(json.dumps(dict(command=command,passed=True,output=args.get('output')),indent=2))


if __name__=='__main__':main()
