# -*- coding: utf-8 -*-
"""
Gappy POD baseline, step 4: headline comparison figure - PINN vs Gappy POD,
region by region, same metric. The whole point of building the POD baseline.

Usage:
    python plot_comparison.py \
        --Pinn ../../runs/E3_freestream_.../regional_evaluation_v2.txt R1 \
        --Pinn ../../runs/R2_bvf_only_.../regional_evaluation_v2.txt R2 \
        --PodR 3 --PodR 8
"""
import argparse
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REGIONS = ['near-cylinder', 'near-wake', 'far-wake', 'other (upstream/off-axis)', 'whole domain']


def parse_pinn_table(path):
    """Parse a src/pressure_only/evaluate_regions.py-style table into {region: (E_u,E_v,E_p)}."""
    out = {}
    with open(path) as f:
        for line in f:
            for region in REGIONS:
                if line.strip().startswith(region):
                    rest = line[len(region):].split()
                    # rest = [n_nodes, E_u, E_v, E_p]
                    out[region] = tuple(float(x) for x in rest[1:4])
    return out


def parse_pod_table(path, r):
    """Parse this repo's gappy_pod evaluate_regions.py output for one r block."""
    out = {}
    with open(path) as f:
        lines = f.readlines()
    in_block = False
    for line in lines:
        if line.strip() == 'r = %d POD modes' % r:
            in_block = True
            continue
        if in_block and line.strip().startswith('r = '):
            break
        if in_block:
            for region in REGIONS:
                if line.strip().startswith(region):
                    rest = line[len(region):].split()
                    out[region] = tuple(float(x) for x in rest[1:4])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Pinn', nargs=2, action='append', default=[],
                         metavar=('PATH', 'LABEL'), help='PINN regional_evaluation(_v2).txt path + label. Repeatable.')
    parser.add_argument('--PodEval', default='pod_regional_evaluation.txt')
    parser.add_argument('--PodR', type=int, action='append', default=[],
                         help='POD r value(s) to include. Repeatable.')
    parser.add_argument('--Out', default='pinn_vs_pod_comparison.png')
    args = parser.parse_args()

    if not args.PodR:
        args.PodR = [3, 8]

    series = {}
    for path, label in args.Pinn:
        series['PINN ' + label] = parse_pinn_table(path)
    for r in args.PodR:
        series['Gappy POD r=%d' % r] = parse_pod_table(args.PodEval, r)

    field_names = ['E_u', 'E_v', 'E_p']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    x = np.arange(len(REGIONS))
    width = 0.8 / len(series)

    for fi, field in enumerate(field_names):
        ax = axes[fi]
        for si, (name, table) in enumerate(series.items()):
            vals = [table.get(region, (np.nan, np.nan, np.nan))[fi] for region in REGIONS]
            ax.bar(x + si * width, vals, width, label=name)
        ax.set_xticks(x + width * (len(series) - 1) / 2)
        ax.set_xticklabels([r.replace(' (upstream/off-axis)', '\n(other)') for r in REGIONS], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('relative L2 error (%s)' % field)
        ax.set_title(field)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=1)
        if fi == 0:
            ax.legend(fontsize=8)

    fig.suptitle('PINN vs Gappy POD - same 32 wall-pressure taps, same regions, same metric')
    plt.tight_layout()
    plt.savefig(args.Out, dpi=150)
    print('Saved comparison figure to', args.Out)

    print('\nParsed series:')
    for name, table in series.items():
        print(' ', name, table)


if __name__ == '__main__':
    main()
