#!/usr/bin/env python3
"""Figura 1 (system overview) sa brojevima koji se CITAJU iz merenja.

Zasto postoji: figura koja je do sada bila u radu nema reproducibilan izvor.
`fig_system_overview.tex` je starija TikZ verzija u boji i ne odgovara crno-beloj
slici u radu, a raniji opis figure nosi zastarele brojeve (694 labela,
95,36 % MSD, 415K tok/s). Zbog toga je figura zaostala za tabelama kroz dva
kruga remerenja. Ovde su svi brojevi izvedeni iz JSON-ova, pa vise ne mogu da
se raziđu sa Tabelom 8 i Sekcijom 4.2.

Ulaz:  NLTK Treniranje/models_v2p_pnpa/training_info.json   (broj MSD klasa)
       results/benchmark_optimized_v2p.json                 (propusnost, RSS, load)
       results/final_system_eval_v2p.json                   (tacnost pod gold tagovima)
Izlaz: results/figures/fig_system_overview_v2p.png (1312x816, kao slika u radu)

Provera: brojevi se stampaju da bi se uporedili sa radom.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'mathtext.fontset': 'dejavusans',
})

BASE = Path(__file__).resolve().parent.parent
PROJECT = BASE.parent
RESULTS = BASE / 'results'
FIG_DIR = RESULTS / 'figures'
OUT = FIG_DIR / 'fig_system_overview_v2p.png'

# Slika koju menjamo u DOCX-u je 1312x816; zadrzavamo iste dimenzije da se
# prelom u dokumentu ne pomeri.
DPI = 150
W_PX, H_PX = 1312, 816

BLACK = '#000000'
GRAYFILL = '#EDEDED'
GRAYTEXT = '#444444'


def load_numbers():
    ti = json.loads((PROJECT / 'NLTK Treniranje' / 'models_v2p_pnpa'
                     / 'training_info.json').read_text())
    bench = json.loads((RESULTS / 'benchmark_optimized_v2p.json').read_text())
    ev = json.loads((RESULTS / 'final_system_eval_v2p.json').read_text())
    n = {
        'classes': ti['classes'],
        'batch_k': bench['end_to_end_parallel']['throughput_mean'] / 1000,
        'latency': bench['end_to_end_single']['latency_ms_mean'],
        'rss': bench['peak_rss_gb']['parent'],
        'load': round(sum(bench['load_time_s'].values()), 2),
        'gold_ud': ev['UD-SET (news)']['gold'],
    }
    print('Brojevi na figuri (uporediti sa radom):')
    print(f"  MSD klasa (Sekcija 4.2, Tabela nema)  {n['classes']}")
    print(f"  batch propusnost (Tabela 8)           {n['batch_k']:.0f}K tok/s")
    print(f"  latencija (Tabela 8)                  {n['latency']} ms/rec")
    print(f"  peak RSS (Tabela 8)                   {n['rss']} GB")
    print(f"  load (Tabela 8)                       {n['load']} s")
    print(f"  tacnost pod gold tagovima (Tabela 2)  {n['gold_ud']} %")
    return n


def box(ax, x, y, w, h, lines, *, dashed=False, fill='white', lw=1.4):
    """Zaobljen pravougaonik sa vise redova teksta; (x, y) je centar."""
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                       boxstyle='round,pad=0.004,rounding_size=0.012',
                       linewidth=lw, edgecolor=BLACK, facecolor=fill,
                       linestyle=(0, (4, 2.5)) if dashed else 'solid',
                       zorder=3)
    ax.add_patch(p)
    n = len(lines)
    for i, (txt, size, weight) in enumerate(lines):
        ty = y + h / 2 - h * (i + 0.5) / n
        ax.text(x, ty, txt, ha='center', va='center', fontsize=size,
                fontweight=weight, color=BLACK, zorder=4)


def arrow(ax, x1, y1, x2, y2, label=None, dashed=False, label_dx=0.012):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='-|>,head_width=0.22,head_length=0.5',
                                color=BLACK, linewidth=1.3,
                                linestyle=(0, (4, 2.5)) if dashed else 'solid',
                                shrinkA=0, shrinkB=0), zorder=2)
    if label:
        ax.text((x1 + x2) / 2 + label_dx, (y1 + y2) / 2, label,
                ha='left', va='center', fontsize=8.5, style='italic',
                color=BLACK, zorder=4)


def brace(ax, y_top, y_bot, x, label):
    """Viticasta zagrada levo od kolone, sa oznakom faze."""
    ax.annotate(label, xy=(x, (y_top + y_bot) / 2),
                xytext=(x - 0.052, (y_top + y_bot) / 2),
                ha='right', va='center', fontsize=9, color=BLACK,
                arrowprops=dict(arrowstyle='-[,widthB=%.2f,lengthB=0.35' %
                                ((y_top - y_bot) * 26),
                                color=BLACK, linewidth=1.1))


def main():
    n = load_numbers()

    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    xm, xr = 0.375, 0.775          # centar glavne kolone i kolone resursa
    wm, wr = 0.40, 0.355           # sirine
    B, R = 'bold', 'normal'

    # --- glavna kolona ---
    box(ax, xm, 0.945, 0.30, 0.075,
        [('Raw text', 11, B), ('(possibly without diacritics)', 8.5, R)],
        fill=GRAYFILL)
    box(ax, xm, 0.815, wm, 0.085,
        [('Tokenizer', 11, B), ('sentence splitting · word tokenization', 8.5, R)])
    box(ax, xm, 0.635, wm, 0.105,
        [('POS Tagger', 11, B), ('averaged perceptron', 8.5, R),
         (f'MSD tags (MULTEXT-East, {n["classes"]} classes)', 8.5, R)])
    box(ax, xm, 0.445, wm, 0.105,
        [('Diacritics Restoration', 11, B), (r'ASCII $\rightarrow$ diacritical form', 8.5, R),
         ('POS-guided disambiguation', 8.5, R)])
    box(ax, xm, 0.255, wm, 0.095,
        [('Cascade Lemmatizer', 11, B),
         (r'MSD exact $\rightarrow$ POS lookup $\rightarrow$ suffix heuristic '
          r'$\rightarrow$ identity', 7.6, R)])
    box(ax, xm, 0.085, 0.30, 0.075,
        [('Lemmatized output', 10.5, B), (r'$\langle$word, MSD, lemma$\rangle$', 8.5, R)],
        fill=GRAYFILL)

    # --- strelice glavnog toka ---
    arrow(ax, xm, 0.9075, xm, 0.858, 'text')
    arrow(ax, xm, 0.7725, xm, 0.688, 'tokens')
    arrow(ax, xm, 0.5825, xm, 0.498, r'$\langle w,\, m \rangle$')
    arrow(ax, xm, 0.3925, xm, 0.303, r"$\langle w',\, m \rangle$")
    arrow(ax, xm, 0.2075, xm, 0.123, r'$\langle w,\, m,\, \hat{l} \rangle$')

    # --- resursi desno (isprekidane kutije) ---
    box(ax, xr, 0.700, wr, 0.095,
        [('POS Model', 10.5, B), ('pre-trained on srWaC', 8.5, R),
         ('fine-tuned on gold corpora', 8.5, R)], dashed=True)
    box(ax, xr, 0.490, wr, 0.085,
        [('srLex v1.3 + expanded supplement', 9.2, B),
         ('1.86M word forms · 175K lemmas', 8.5, R)], dashed=True)
    box(ax, xr, 0.300, wr, 0.085,
        [('Corrections', 10.5, B),
         ('ekavization · 148 reviewed lemma fixes', 8.2, R)], dashed=True)

    arrow(ax, xr - wr / 2, 0.700, xm + wm / 2, 0.655, dashed=True)
    arrow(ax, xr - wr / 2, 0.490, xm + wm / 2, 0.462, dashed=True)
    arrow(ax, xr - wr / 2, 0.470, xm + wm / 2, 0.272, dashed=True)
    arrow(ax, xr - wr / 2, 0.300, xm + wm / 2, 0.250, dashed=True)

    # --- performanse (dole desno) ---
    box(ax, xr, 0.105, wr, 0.135,
        [('Performance (CPU, Apple M2)', 9.5, B),
         (f'{n["gold_ud"]:.2f}% accuracy (UD test, gold MSD)', 8.5, R),
         (f'{n["batch_k"]:.0f}K tokens/s batch · {n["latency"]:.2f} ms/sentence', 8.5, R),
         (f'{n["rss"]:.2f} GB peak memory · {n["load"]:.2f} s model load', 8.5, R)],
        fill=GRAYFILL)
    arrow(ax, xm + 0.15, 0.085, xr - wr / 2, 0.105, dashed=True)

    # --- viticaste zagrade sa fazama ---
    for y_top, y_bot, label in [(0.858, 0.773, 'Stage 1'),
                                (0.688, 0.583, 'Stage 2'),
                                (0.498, 0.393, 'Stage 3'),
                                (0.303, 0.208, 'Stage 4')]:
        brace(ax, y_top, y_bot, xm - wm / 2 - 0.012, label)

    fig.savefig(OUT, dpi=DPI, facecolor='white')
    plt.close(fig)

    import struct
    dims = struct.unpack('>II', OUT.read_bytes()[16:24])
    print(f'\nSacuvano: {OUT}  ({dims[0]}x{dims[1]} px)')
    if dims != (W_PX, H_PX):
        print(f'PAZNJA: dimenzije nisu {W_PX}x{H_PX}, zamena u DOCX-u ce skalirati')


if __name__ == '__main__':
    main()
