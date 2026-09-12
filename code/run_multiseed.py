#!/usr/bin/env python3
"""Run repeated PrivNet-Krylov simulations and summarize robustness."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from privnet_krylov import ExperimentConfig, run


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[1,2,3,4,5,6,7])
    ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1]/'outputs'/'multiseed')
    args=ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for seed in args.seeds:
        d=run(args.out/f'seed_{seed}', ExperimentConfig(seed=seed))
        row={'seed':seed,'relu_test':d['plaintext_relu_accuracy']['test']}
        for r in d['activation_tradeoff']:
            row[f'd{r["degree"]}_acc']=r['test_accuracy_simulated_ckks']
            row[f'd{r["degree"]}_cert']=r['certified_fraction_test_nodes']
        rows.append(row)
    keys=[k for k in rows[0] if k!='seed']
    summary={}
    for k in keys:
        v=np.array([r[k] for r in rows],float)
        summary[k]={'mean':float(v.mean()),'std':float(v.std(ddof=1)),'min':float(v.min()),'max':float(v.max())}
    root=args.out.parent
    (root/'multiseed_summary.json').write_text(json.dumps({'n_seeds':len(rows),'rows':rows,'summary':summary},indent=2))
    with (root/'multiseed_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

if __name__=='__main__': main()
