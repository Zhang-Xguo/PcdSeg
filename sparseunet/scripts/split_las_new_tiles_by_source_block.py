#!/usr/bin/env python3
"""Split prepared las_new tiles by original source Block before making lists."""
import argparse, json, shutil
from pathlib import Path
import numpy as np
import laspy

ap = argparse.ArgumentParser(); ap.add_argument('--data-root', type=Path, required=True); a=ap.parse_args()
root=a.data_root; report=json.loads((root/'las_new_target_val_final_report.json').read_text())
orig=[Path(x) for x in report['original_las']]
mins=np.min([laspy.open(x).header.mins for x in orig],axis=0)
bounds=[(laspy.open(x).header.mins-mins, laspy.open(x).header.maxs-mins) for x in orig]
all_paths=json.loads((root/'val_las_new.json').read_text()); train=[]; hold=[]
for rel in all_paths:
 src=root/rel; name=Path(rel).name; data={k:np.load(src/(k+'.npy')) for k in ('coord','color','segment','label_source','supervision_weight')}
 assigned=np.full(len(data['coord']),-1,dtype=np.int16)
 for bid,(lo,hi) in enumerate(bounds):
  c=data['coord']; m=(c[:,0]>=lo[0])&(c[:,0]<=hi[0])&(c[:,1]>=lo[1])&(c[:,1]<=hi[1]); assigned[m]=bid
 if (assigned<0).any(): raise RuntimeError(f'unassigned points in {name}')
 for bid in range(len(bounds)):
  ids=np.flatnonzero(assigned==bid)
  if len(ids)==0: continue
  out=root/'val_final'/f'las_new_target_b{bid:02d}__{name.split("__",1)[1]}'; out.mkdir(exist_ok=True)
  for k,v in data.items(): np.save(out/(k+'.npy'),v[ids])
  relout='val_final/'+out.name
  (train if bid<7 else hold).append(relout)
(root/'train_las_new_blocks.json').write_text(json.dumps(sorted(train),indent=2)+'\n')
(root/'val_las_new_blocks_holdout.json').write_text(json.dumps(sorted(hold),indent=2)+'\n')
fixed=[p for p in json.loads((root/'val_fixed.json').read_text()) if not p.startswith('val_final/las_new_target__')]
(root/'val_fixed_las_blocks.json').write_text(json.dumps(sorted(fixed+hold),indent=2)+'\n')
print(json.dumps({'train_tiles':len(train),'holdout_tiles':len(hold),'train_blocks':'0-6','holdout_blocks':'7-13'}))
