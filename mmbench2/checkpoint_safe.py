from pathlib import Path
import os,shutil,torch

def save_checkpoint(payload,folder,update):
 folder=Path(folder);latest=folder/'latest.pt';tmp=folder/f'.latest-{os.getpid()}.tmp';torch.save(payload,tmp)
 with tmp.open('rb')as f:os.fsync(f.fileno())
 ck=torch.load(tmp,map_location='cpu',mmap=True,weights_only=False)
 assert ck['optimizer_updates']==update and ck['opt']['state']and ck['rms_state']
 assert ('model'in ck)or all(k in ck for k in ['dynamics','rew_head','policy_head']);del ck
 if latest.exists():
  previous=folder/'latest.previous.pt';q=previous.with_suffix('.pending')
  if q.exists():q.unlink()
  try:os.link(latest,q)
  except OSError:shutil.copy2(latest,q)
  q.replace(previous)
 tmp.replace(latest)
 if update%5000==0 or update==payload['experiment_config']['optimizer_updates']:
  retained=folder/f'update_{update:06d}.pt'
  if not retained.exists():os.link(latest,retained)
