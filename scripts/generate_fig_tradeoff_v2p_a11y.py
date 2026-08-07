#!/usr/bin/env python3
"""Figura 4 (throughput) u verziji citljivoj i bez razlikovanja boja.

Zasto: postojeca figura razlikuje dva sistema SAMO tonom (#3498db naspram
#e74c3c). Simulacija daltonizma pokazuje da trake ostaju razlicite (plavo-ljubicasta
naspram maslinaste), ali je odnos svetline svega 1,21:1, pa u crno-beloj stampi i pri
fotokopiranju postaju gotovo iste. Boja je tu jedini nosilac informacije.

Sta se menja (sadrzaj i brojevi ostaju identicni):
  1. paleta Okabe-Ito, koja je napravljena za sve tipove daltonizma: plava #0072B2
     za predlozeni sistem, narandzasta #E69F00 za CLASSLA-u;
  2. te dve boje se razlikuju i po SVETLINI (odnos ~3,3:1), pa se razlikuju i u
     sivim tonovima;
  3. trake CLASSLA-e dobijaju kosu srafuru, sto je znak nezavisan od boje;
  4. crne ivice traka umesto belih, radi kontrasta na crno-beloj stampi.
Brojevi iznad traka su i dalje tu i sami po sebi nose podatak.

Ulaz i izlaz su isti kao kod generate_fig_tradeoff_v2p.py, samo drugi izlazni stem,
da originalna figura ostane netaknuta.

Izlaz: results/figures/fig_tradeoff_v2p_a11y.{png,pdf}
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    'hatch.linewidth': 0.8,
})

BASE = Path(__file__).resolve().parent.parent
RESULTS = BASE / 'results'
FIG_DIR = RESULTS / 'figures'
OURS_JSON = RESULTS / 'benchmark_optimized_v2p.json'
CLASSLA_JSON = RESULTS / 'benchmark_classla_scenarios_rerun.json'
OUT_STEM = FIG_DIR / 'fig_tradeoff_v2p_a11y'

BLUE = '#0072B2'     # Okabe-Ito blue
ORANGE = '#E69F00'   # Okabe-Ito orange
TEXT = '#2c3e50'


def rel_lum(hexs):
    c = [int(hexs[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4) for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def main():
    o = json.loads(OURS_JSON.read_text())
    c = json.loads(CLASSLA_JSON.read_text())

    lb, lo = rel_lum(BLUE), rel_lum(ORANGE)
    ratio = (max(lb, lo) + 0.05) / (min(lb, lo) + 0.05)
    print(f'Odnos svetline {BLUE} : {ORANGE} = {ratio:.2f} : 1 '
          f'(stara paleta je imala 1,21 : 1)')

    scenarios = ['Batch', 'Single-stream', 'Lemmatization-only\n(MSD available)']
    ours = [o['end_to_end_parallel']['throughput_mean'],
            o['end_to_end_single']['throughput_mean'],
            o['lemma_only_compact']['throughput_mean']]
    ours_err = [o['end_to_end_parallel']['throughput_std'],
                o['end_to_end_single']['throughput_std'],
                o['lemma_only_compact']['throughput_std']]
    classla = [c['batch_tok_s'][0], c['single_stream_tok_s'][0], None]
    classla_err = [c['batch_tok_s'][1], c['single_stream_tok_s'][1], None]

    print('\nBrojevi na figuri (uporediti sa Tabelom 8):')
    for s, v, e in zip(scenarios, ours, ours_err):
        print(f'  nas      {s.splitlines()[0]:18s} {v:,} +/- {e:,}')
    for s, v, e in zip(scenarios, classla, classla_err):
        if v is not None:
            print(f'  CLASSLA  {s.splitlines()[0]:18s} {v:,} +/- {e:,}')

    x = np.arange(len(scenarios))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    bars1 = ax.bar(x - width / 2, ours, width, yerr=ours_err, capsize=3,
                   label='Proposed', color=BLUE, edgecolor='black', linewidth=0.6)
    cl_x = [xi + width / 2 for xi, v in zip(x, classla) if v is not None]
    cl_v = [v for v in classla if v is not None]
    cl_e = [e for e in classla_err if e is not None]
    bars2 = ax.bar(cl_x, cl_v, width, yerr=cl_e, capsize=3,
                   label='CLASSLA-Stanza', color=ORANGE, edgecolor='black',
                   linewidth=0.6, hatch='//')

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
                ha='center', fontsize=8, color=TEXT)
    for bar, v in zip(bars2, cl_v):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.35, f'{v:,}',
                ha='center', fontsize=8, color=TEXT)
    ax.text(x[2] + width / 2, 700, 'n/a', ha='center', fontsize=8,
            color='#6b7b8c', style='italic')

    plt.tight_layout()
    plt.savefig(str(OUT_STEM) + '.pdf')
    plt.savefig(str(OUT_STEM) + '.png')
    plt.close()
    print(f'\nSacuvano: {OUT_STEM}.png i .pdf')


if __name__ == '__main__':
    main()
