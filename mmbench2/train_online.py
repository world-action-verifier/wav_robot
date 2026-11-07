"""Train online WAV alpha=0.75 on an immutable N50 collection snapshot."""
import argparse
from round_config import activate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['tokenizer','dynamics'])
    p.add_argument('--updates',type=int)
    p.add_argument('--micro_batch',type=int,default=0)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();activate()
    a.setting='wav-cem-alpha075';a.node=0
    if a.updates is None:a.updates=50000 if a.stage=='tokenizer' else 30000
    import train_stage
    from online_stream import RawStreamDataset
    train_stage.RawStreamDataset=RawStreamDataset
    train_stage.run(a)

if __name__=='__main__':main()
