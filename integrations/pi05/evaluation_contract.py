"""Paired plans and complete-result aggregation, usable without GPU libraries."""
import hashlib
import json
from pathlib import Path


def read_plan(path):
    entries = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not entries or len({e['episode_id'] for e in entries}) != len(entries):
        raise ValueError('Nonempty plan with unique episode IDs required.')
    for e in entries:
        if not e['episode_id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in e['episode_id']):
            raise ValueError('Episode IDs must be safe filename components.')
        for key in ['room','task_id','instruction','environment_seed','policy_seed','state_key','state_hash','slot']:
            if key not in e:
                raise ValueError('Missing paired case field: ' + key)
    return entries


def bind_output(out, plan, checkpoint_id, assets):
    out, assets = Path(out), Path(assets)
    out.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint_id': checkpoint_id, 'plan_sha256': hashlib.sha256(Path(plan).read_bytes()).hexdigest()}
    for name in ['TASKS_30.json','EVAL_DEV_STATES.npz']:
        identity[name+'_sha256'] = hashlib.sha256((assets/name).read_bytes()).hexdigest()
    p = out/'IDENTITY.json'
    if p.exists() and json.loads(p.read_text()) != identity:
        raise ValueError('Evaluation directory belongs to another checkpoint, plan or environment.')
    p.write_text(json.dumps(identity, indent=2)+'\n')
    return identity


def summarize(entries, out, checkpoint_id):
    rows = []
    cells = {}
    for e in entries:
        key = (e['room'],e['task_id'])
        cell = cells.setdefault(key, dict(room=key[0],task_id=key[1],expected=0,completed=0,success=0,technical_errors=0))
        cell['expected'] += 1
        p = Path(out)/'episodes'/(e['episode_id']+'.json')
        if not p.exists():
            continue
        row = json.loads(p.read_text())
        if row['checkpoint_id'] != checkpoint_id or row['episode_id'] != e['episode_id']:
            raise ValueError('Mixed result identity.')
        if row.get('valid'):
            if not isinstance(row.get('success'),bool) or row.get('steps',0)<=0:
                raise ValueError('Malformed completed episode.')
            cell['completed'] += 1; cell['success'] += int(row['success']); rows.append(row)
        else:
            cell['technical_errors'] += 1
    for c in cells.values():
        c['success_rate'] = c['success']/c['completed'] if c['completed'] else None
    complete = len(rows)==len(entries)
    return dict(status='COMPLETE' if complete else 'PARTIAL',expected=len(entries),completed=len(rows),
                success=sum(r['success'] for r in rows),success_rate=(sum(r['success'] for r in rows)/len(rows) if rows else None),
                final_success_rate=(sum(r['success'] for r in rows)/len(entries) if complete else None),
                by_cell=sorted(cells.values(),key=lambda c:(c['room'],c['task_id'])))
