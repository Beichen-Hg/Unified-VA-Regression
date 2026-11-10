"""Portable JSON/JSONL and atomic run records."""
from datetime import datetime, timezone
from pathlib import Path
import json,os


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def read_rows(path):
    path=Path(path)
    if path.suffix=='.jsonl':
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    value=read(path)
    if not isinstance(value,list):raise ValueError('Expected a JSON list or JSONL file')
    return value


def write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix=='.jsonl':
        temporary=path.with_name(path.name+'.tmp')
        with temporary.open('w',encoding='utf-8',newline='\n') as stream:
            for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        os.replace(temporary,path)
    else:write(path,rows)
