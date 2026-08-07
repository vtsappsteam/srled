#!/usr/bin/env python3
"""
Build expanded_supplement_v2.json = existing expanded_supplement.json
(SETimes.SR 2.0 train+dev + ReLDI train, built by expand_dictionary.py)
+ pairs mined from the SrpKor4Tagging TRAIN portion.

SrpKor4Tagging (HF: jerteh/SrpKor4Tagging) has no official split; the
evaluation protocol (evaluate_three_testsets.load_srpkor_test) uses the
LAST 10% of rows as the held-out test set. This script therefore mines
ONLY the FIRST 90% of rows (the complement), so no test sentence ever
enters the dictionary.

Filters are identical to expand_dictionary.py:
  - word form (lowercased) not present in srLex v1.3
  - lemma != '_'
  - len(word) > 1
  - every surviving pair is kept (freq >= 1), frequency = train count

MSD field for SrpKor entries: SrpKor gold is UPOS, not full MULTEXT-East
MSD. Each mined entry stores the SINGLE-CHARACTER coarse tag obtained via
the same UPOS -> MSD-first-letter map used in evaluation (UPOS_TO_MSD).
Inside the lemmatizer these entries behave as follows: the exact-MSD
lookup matches only when the query MSD is itself that single character
(the SrpKor evaluation setting); for full-MSD queries (UD/ReLDI) the
entries are reached through the POS index (msd[0]) and the MSD-prefix
fallback, plus the word index. PUNCT tokens are skipped (they are also
excluded from evaluation).

Usage:
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python3 scripts/build_supplement_v2.py
Output: data/expanded_supplement_v2.json
"""

import gzip
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
BASE = SCRIPT_DIR.parent.parent.parent  # .../NLP-POS-Tagging
SRLEX = BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz'
EXISTING = SCRIPT_DIR.parent / 'data' / 'expanded_supplement.json'
OUT = SCRIPT_DIR.parent / 'data' / 'expanded_supplement_v2.json'

UPOS_TO_MSD = {
    'NOUN': 'N', 'PROPN': 'N', 'VERB': 'V', 'AUX': 'V',
    'ADJ': 'A', 'ADV': 'R', 'PRON': 'P', 'DET': 'P',
    'ADP': 'S', 'CCONJ': 'C', 'SCONJ': 'C', 'NUM': 'M',
    'PART': 'Q', 'INTJ': 'I', 'PUNCT': 'Z', 'X': 'X',
}


def load_srlex_words(path):
    words = set()
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                words.add(parts[0].lower())
    return words


def main():
    print('1. srLex vocabulary...', flush=True)
    srlex_words = load_srlex_words(SRLEX)
    print(f'   {len(srlex_words):,} forms')

    print('2. SrpKor4Tagging train portion (first 90% of rows)...', flush=True)
    ds = load_dataset('jerteh/SrpKor4Tagging')['train']
    n = len(ds)
    test_size = n // 10
    train_end = n - test_size  # rows [0, train_end) = train; test = last 10%
    print(f'   rows total={n}, train={train_end}, held-out test={test_size}')

    tokens = ds['token']
    lemmas = ds['lemma']
    uposes = ds['ud']

    new_pairs = Counter()
    n_tok = 0
    for idx in range(train_end):
        for w, lemma, upos in zip(tokens[idx], lemmas[idx], uposes[idx]):
            if upos == 'PUNCT':
                continue
            n_tok += 1
            w_lower = w.lower()
            if w_lower not in srlex_words and lemma != '_' and len(w) > 1:
                msd = UPOS_TO_MSD.get(upos, 'X')
                new_pairs[(w_lower, lemma.lower(), msd)] += 1
    print(f'   train tokens (non-PUNCT): {n_tok:,}')
    print(f'   OOV (word, lemma, coarse-MSD) pairs: {len(new_pairs):,}')
    print('   Top examples:')
    for (w, l, m), f in new_pairs.most_common(10):
        print(f'     {w:<25} -> {l:<25} [{m}] freq={f}')

    print('3. Merging with existing supplement...', flush=True)
    with open(EXISTING, encoding='utf-8') as f:
        merged = {w: [list(e) for e in entries]
                  for w, entries in json.load(f).items()}
    n_old_words = len(merged)
    n_old_entries = sum(len(v) for v in merged.values())

    added_entries = summed = 0
    for (w, lemma, msd), freq in new_pairs.items():
        entries = merged.setdefault(w, [])
        for e in entries:
            if e[0] == lemma and e[1] == msd:
                e[2] += freq
                summed += 1
                break
        else:
            entries.append([lemma, msd, freq])
            added_entries += 1

    n_words = len(merged)
    n_entries = sum(len(v) for v in merged.values())
    print(f'   existing: {n_old_words:,} words / {n_old_entries:,} entries')
    print(f'   v2:       {n_words:,} words / {n_entries:,} entries '
          f'(+{added_entries:,} new, {summed} freq-merged)')

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=0)
    print(f'   Saved: {OUT} ({OUT.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
