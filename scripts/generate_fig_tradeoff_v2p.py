#!/usr/bin/env python3
"""Figura 4 (throughput po scenariju) sa brojevima iz retreniranog tagera.

Razlika u odnosu na repo/scripts/generate_figures.py::fig_tradeoff: brojevi se
CITAJU iz merenja umesto da budu hardkodovani. Hardkodovane vrednosti u toj
funkciji su preziveli dva kruga remerenja i figura je zaostajala za tabelom
(31. jul: tabela 180.223, figura jos 163.441).

Ulaz:  results/benchmark_optimized_v2p.json      (nas sistem)
       results/benchmark_classla_scenarios_rerun.json (CLASSLA)
Izlaz: results/figures/fig_tradeoff_v2p.{png,pdf}  (original netaknut)

Provera: brojevi na figuri se stampaju da bi se uporedili sa Tabelom 8.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Isti stil kao ostale figure u radu (Springer: sans-serif, 8-12 pt).
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIG_DIR = RESULTS / 'figures'
OURS_JSON = RESULTS / 'benchmark_optimized_v2p.json'
CLASSLA_JSON = RESULTS / 'benchmark_classla_scenarios_rerun.json'
OUT_STEM = FIG_DIR / 'fig_tradeoff_v2p'


def main():
    o = json.loads(OURS_JSON.read_text())
    c = json.loads(CLASSLA_JSON.read_text())

    scenarios = ['Batch', 'Single-stream', 'Lemmatization-only\n(MSD available)']
    ours = [o['end_to_end_parallel']['throughput_mean'],
            o['end_to_end_single']['throughput_mean'],
            o['lemma_only_compact']['throughput_mean']]
    ours_err = [o['end_to_end_parallel']['throughput_std'],
                o['end_to_end_single']['throughput_std'],
                o['lemma_only_compact']['throughput_std']]
    classla = [c['batch_tok_s'][0], c['single_stream_tok_s'][0], None]
    classla_err = [c['batch_tok_s'][1], c['single_stream_tok_s'][1], None]

    print('Brojevi na figuri (uporediti sa Tabelom 8):')
    for s, v, e in zip(scenarios, ours, ours_err):
        print(f'  nas      {s.splitlines()[0]:18s} {v:,} +/- {e:,}')
    for s, v, e in zip(scenarios, classla, classla_err):
        if v is not None:
            print(f'  CLASSLA  {s.splitlines()[0]:18s} {v:,} +/- {e:,}')

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars1 = ax.bar(x - width / 2, ours, width, yerr=ours_err, capsize=3,
                   label='Proposed', color='#3498db', edgecolor='white')
    cl_x = [xi + width / 2 for xi, v in zip(x, classla) if v is not None]
    cl_v = [v for v in classla if v is not None]
    cl_e = [e for e in classla_err if e is not None]
    bars2 = ax.bar(cl_x, cl_v, width, yerr=cl_e, capsize=3,
                   label='CLASSLA-Stanza', color='#e74c3c', edgecolor='white')

    ax.set_yscale('log')
    ax.set_ylabel('Throughput (tokens/sec, log scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylim(500, 8_000_000)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_title('Throughput by Deployment Scenario (CPU only)')
    ax.grid(True, axis='y', alpha=0.3)

    for bar, v in zip(bars1, ours):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.35, f'{v:,}',
                ha='center', fontsize=8, color='#2c3e50')
    for bar, v in zip(bars2, cl_v):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.35, f'{v:,}',
                ha='center', fontsize=8, color='#2c3e50')
    # CLASSLA ne moze da preskoci svoj neuronski tager, pa nema lemma-only stubac
    ax.text(x[2] + width / 2, 700, 'n/a', ha='center', fontsize=8,
            color='#95a5a6', style='italic')

    plt.tight_layout()
    plt.savefig(str(OUT_STEM) + '.pdf')
    plt.savefig(str(OUT_STEM) + '.png')
    plt.close()
    print(f'\nSacuvano: {OUT_STEM}.png i .pdf')


if __name__ == '__main__':
    main()
