#!/usr/bin/env python3
"""
Data-driven lemma-correction mining from gold TRAIN splits only.

The lemmatizer's LEMMA_CORRECTIONS table (17 manual entries) globally maps
a selected lemma to a corrected one. This script mines candidates for that
table: system-output lemmas (lemmatizer with GOLD tags, srLex + expanded
supplement v2, the 17 manual corrections active) whose gold lemma
SYSTEMATICALLY disagrees in the training data.

Train sources (no test split touches any resource):
  - UD Serbian-SET train        (/tmp/UD_Serbian-SET/sr_set-ud-train.conllu)
  - SETimes.SR 2.0 train + dev  (/tmp/SETimes_SR_2.0/{train,dev}.conllu)
  - ReLDI train                 (contained_in_datasets split)
  - SrpKor4Tagging first 90% of rows (test = last 10%; UPOS -> coarse char)
  Napomena: SETimes.SR 2.0 train is the same corpus/split as UD train, so
  news evidence is counted roughly twice; this affects fix and break counts
  symmetrically and is documented in the report.

For every train token: key = system lemma (lowercased), value = gold lemma
(lowercased). For each system lemma with >= MIN_CAND occurrences whose
dominant gold lemma differs from it with >= CAND_SHARE agreement, emit a
candidate mapping system_lemma -> dominant_gold_lemma.

APPLIED (conservative) subset, every condition mechanical:
  - total occurrences >= 5
  - dominant-gold agreement >= 95%
  - ZERO train breaks: gold lemma NEVER equals the source lemma when the
    system outputs it (the mapping cannot break a single train token)
  - in every corpus where the source occurs, the dominant gold there is the
    same target (no cross-corpus convention conflict)

Usage:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/mine_lemma_corrections.py
Outputs:
    results/lemma_correction_candidates.csv  (full list, for review)
    data/lemma_corrections_v2.csv            (17 manual + applied subset)
"""

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402
import scispace_experiments  # noqa: E402,F401  (fixes base.BASE/SRLEX paths)
from expand_dictionary import load_reldi_splits  # noqa: E402

EXTRA_V2 = SCRIPT_DIR.parent / 'data' / 'expanded_supplement_v2.json'
CAND_CSV = SCRIPT_DIR.parent / 'results' / 'lemma_correction_candidates.csv'
ORIG_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections.csv'
OUT_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections_v2.csv'

MIN_CAND, CAND_SHARE = 3, 0.90   # candidate list (for review)
MIN_APPLY, APPLY_SHARE = 5, 0.95  # applied subset (+ zero breaks + per-corpus)

UPOS_TO_MSD = base.UPOS_TO_MSD


def load_srpkor_train():
    ds = load_dataset('jerteh/SrpKor4Tagging')['train']
    n = len(ds)
    train_end = n - n // 10  # complement of load_srpkor_test's last 10%
    tokens, lemmas, uposes = ds['token'], ds['lemma'], ds['ud']
    sents = []
    for idx in range(train_end):
        sent = [(w, l, UPOS_TO_MSD.get(u, 'X'))
                for w, l, u in zip(tokens[idx], lemmas[idx], uposes[idx])
                if u != 'PUNCT']
        if sent:
            sents.append(sent)
    return sents


def main():
    print('Ucitavanje train izvora...', flush=True)
    sources = {}
    sources['UD_train'] = base.load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-train.conllu')
    sources['SET2_train_dev'] = (base.load_conllu('/tmp/SETimes_SR_2.0/train.conllu')
                                 + base.load_conllu('/tmp/SETimes_SR_2.0/dev.conllu'))
    reldi_train, _ = load_reldi_splits(
        str(base.BASE / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup'))
    sources['ReLDI_train'] = reldi_train
    sources['SrpKor_train'] = load_srpkor_train()
    for name, sents in sources.items():
        print(f'  {name}: {len(sents)} sent, '
              f'{sum(len(s) for s in sents):,} tokens', flush=True)

    print('Lematizator (srLex + expanded supplement v2, 17 rucnih korekcija)...',
          flush=True)
    lem = base.Lemmatizer(base.SRLEX, str(EXTRA_V2))

    # dist[sys_lemma][gold_lemma] = count; per_corpus[sys][corpus][gold] = count
    dist = defaultdict(Counter)
    per_corpus = defaultdict(lambda: defaultdict(Counter))
    forms = defaultdict(Counter)
    for cname, sents in sources.items():
        print(f'Lematizacija {cname}...', flush=True)
        for sent in sents:
            for w, gl, msd in sent:
                if not gl or gl == '_':
                    continue
                sysl = (lem.lemmatize(w, msd) or '').lower()
                gl = gl.lower()
                dist[sysl][gl] += 1
                per_corpus[sysl][cname][gl] += 1
                forms[sysl][w.lower()] += 1

    manual = {}
    with open(ORIG_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            manual[row['source_lemma']] = row['corrected_lemma']

    print('Kandidati...', flush=True)
    candidates = []
    for sysl, golds in dist.items():
        total = sum(golds.values())
        if total < MIN_CAND or not sysl:
            continue
        (top_gold, top_n), = golds.most_common(1)
        if top_gold == sysl or top_n / total < CAND_SHARE:
            continue
        breaks = golds.get(sysl, 0)
        # per-corpus consistency: dominant gold in every corpus == top_gold
        corpus_counts = {c: dict(g) for c, g in per_corpus[sysl].items()}
        consistent = all(g.most_common(1)[0][0] == top_gold
                         for g in per_corpus[sysl].values())
        apply = (total >= MIN_APPLY and top_n / total >= APPLY_SHARE
                 and breaks == 0 and consistent
                 and sysl not in manual)
        candidates.append({
            'source_lemma': sysl,
            'target_lemma': top_gold,
            'total': total,
            'agreement': round(top_n / total, 3),
            'train_breaks_gold_eq_source': breaks,
            'per_corpus_consistent': consistent,
            'applied': apply,
            'in_manual_17': sysl in manual,
            'corpora': '; '.join(f'{c}:{sum(g.values())}' for c, g in
                                 sorted(per_corpus[sysl].items())),
            'example_forms': ' '.join(w for w, _ in forms[sysl].most_common(5)),
            'gold_distribution': '; '.join(f'{g}:{n}' for g, n in
                                           golds.most_common(5)),
        })
    candidates.sort(key=lambda c: -c['total'])
    applied = [c for c in candidates if c['applied']]
    print(f'  kandidata (>= {MIN_CAND} pojava, >= {CAND_SHARE:.0%}): '
          f'{len(candidates)}')
    print(f'  primenjeno (>= {MIN_APPLY}, >= {APPLY_SHARE:.0%}, 0 breaks, '
          f'per-corpus konzistentno): {len(applied)}')
    for c in applied[:40]:
        print(f"    {c['source_lemma']:<20} -> {c['target_lemma']:<20} "
              f"n={c['total']:<5} agr={c['agreement']:.2f} [{c['corpora']}]")

    CAND_CSV.parent.mkdir(exist_ok=True)
    with open(CAND_CSV, 'w', encoding='utf-8', newline='') as f:
        wcsv = csv.DictWriter(f, fieldnames=list(candidates[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(candidates)
    print(f'Kandidat-lista: {CAND_CSV}')

    with open(ORIG_CSV, encoding='utf-8') as f:
        orig_lines = f.read().rstrip('\n')
    with open(OUT_CSV, 'w', encoding='utf-8') as f:
        f.write(orig_lines + '\n')
        for c in applied:
            f.write(f"{c['source_lemma']},{c['target_lemma']},"
                    f"data-driven from train splits "
                    f"(n={c['total']}, agreement={c['agreement']:.0%})\n")
    print(f'lemma_corrections_v2.csv: {OUT_CSV} '
          f'(17 rucnih + {len(applied)} novih)')


if __name__ == '__main__':
    main()
