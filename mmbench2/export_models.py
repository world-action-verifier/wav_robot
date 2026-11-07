"""Expose immutable final training checkpoints to the evaluation adapters."""
import argparse,os
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--setting',required=True)
    p.add_argument('--online',action='store_true')
    args=p.parse_args()
    if args.online:
        from round_config import activate
        activate()
    from config import RUN
    import torch
    out=RUN/'models'/args.setting;out.mkdir(parents=True,exist_ok=True)
    for stage,updates,key in [('tokenizer',50000,'model'),('dynamics',30000,'dynamics')]:
        path=RUN/'training'/args.setting/'seed0'/stage/f'update_{updates:06d}.pt'
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        if state.get('optimizer_updates')!=updates or key not in state:
            raise ValueError(f'Final {stage} checkpoint is incomplete: {path}')
        target=out/(stage+'.pt')
        if target.is_symlink():
            if target.resolve()!=path.resolve():raise FileExistsError(target)
        elif target.exists():raise FileExistsError(target)
        else:target.symlink_to(os.path.relpath(path,out))
        print(target)

if __name__=='__main__':main()
