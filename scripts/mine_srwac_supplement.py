#!/usr/bin/env python3
"""
Mine the full local srWaC chunk (srWaC1.1.01.xml, vertical format) for an
expanded diacritics supplement.

For every surface form in the corpus, count frequencies (skipping foreign
tokens tagged Xf). Then, for every ASCII key whose srLex lookup is empty
(OOV for the main dictionary), choose the most frequent srWaC variant
(including the ASCII-identical variant). An entry is emitted only when a
diacritized variant strictly dominates the ASCII-identical one, so that
restoration never fires against corpus evidence.

Outputs:
  data/srwac_supplement_v2.json  { ascii_key: {word, freq} }   (OOV keys)
  data/srwac_augment_np.json     { ascii_key: {word, freq} }   (proper-noun
      augmentation: keys already in srLex whose CAPITALIZED diacritized
      variant dominates the capitalized ASCII variant in srWaC; consulted
      only when the tagger assigns Np and the token is capitalized)

"""
import os
import sys
import json
import gzip
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from diacritics import strip_diacritics

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
SRWAC_XML = os.environ.get('SRWAC_XML', 'srWaC1.1.01.xml')
SRLEX_PATH = os.path.join(BASE_DIR, 'POS-Aware-Stemmer', 'data', 'srLex_v1.3.gz')
OLD_SUPP = os.path.join(os.path.dirname(__file__), '..', 'data', 'srwac_supplement.json')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'srwac_supplement_v2.json')
OUT_NP = os.path.join(os.path.dirname(__file__), '..', 'data', 'srwac_augment_np.json')

MIN_FREQ = 3          # minimum srWaC frequency for a variant to be trusted
DOMINANCE = 2.0       # diacritized variant must be >= 2x the ASCII-identical freq
DIAC_CHARS = set('čćžšđ')


def is_clean_token(tok):
    """Alphabetic Serbian token, optionally with inner hyphen/digits."""
    if not tok or len(tok) > 40:
        return False
    for ch in tok:
        if not (ch.isalpha() or ch.isdigit() or ch == '-'):
            return False
    return any(ch.isalpha() for ch in tok)


def main():
    print('1. Counting srWaC surface forms...', flush=True)
    freq = Counter()       # lowercase counts (for OOV supplement)
    cap_freq = Counter()   # capitalized-surface counts (for Np augmentation)
    n_lines = 0
    with open(SRWAC_XML, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('<'):
                continue
            n_lines += 1
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 4:
                continue
            surface, msd = parts[0], parts[3]
            if msd == 'Xf' or msd == 'Z':      # foreign tokens, punctuation
                continue
            tok = surface.lower()
            if is_clean_token(tok):
                freq[tok] += 1
                if surface[:1].isupper() and surface[1:].islower():
                    cap_freq[surface] += 1
    print(f'   {n_lines:,} token lines, {len(freq):,} distinct clean forms, '
          f'{len(cap_freq):,} capitalized forms', flush=True)

    print('2. Loading srLex ASCII keys...', flush=True)
    srlex_ascii = set()
    with gzip.open(SRLEX_PATH, 'rt', encoding='utf-8') as f:
        for line in f:
            p = line.split('\t', 1)
            if p:
                srlex_ascii.add(strip_diacritics(p[0]).lower())
    print(f'   {len(srlex_ascii):,} srLex ASCII keys', flush=True)

    print('3. Building OOV variant distributions...', flush=True)
    # ascii_key -> {variant: freq}, only for keys not covered by srLex
    dist = defaultdict(dict)
    for form, n in freq.items():
        if n < MIN_FREQ:
            continue
        ascii_key = strip_diacritics(form)
        if ascii_key in srlex_ascii:
            continue
        dist[ascii_key][form] = n

    supplement = {}
    for ascii_key, variants in dist.items():
        # most frequent variant overall
        best, best_n = max(variants.items(), key=lambda kv: kv[1])
        if not (set(best) & DIAC_CHARS):
            continue  # best variant is the ASCII form itself -> nothing to restore
        identity_n = variants.get(ascii_key, 0)
        if best_n < DOMINANCE * max(identity_n, 1):
            continue  # diacritized variant does not dominate -> unsafe
        supplement[ascii_key] = {'word': best, 'freq': best_n}

    print(f'   v2 supplement: {len(supplement):,} entries', flush=True)

    old = json.load(open(OLD_SUPP))
    kept = sum(1 for k in old if k in supplement)
    print(f'   old supplement: {len(old):,}; covered by v2: {kept:,}', flush=True)
    # merge: keep old entries that v2 missed (they came from the same corpus
    # but possibly different thresholds); v2 wins on conflicts (better stats)
    added_from_old = 0
    for k, v in old.items():
        if k not in supplement:
            supplement[k] = v
            added_from_old += 1
    print(f'   merged in {added_from_old:,} old-only entries -> total {len(supplement):,}', flush=True)

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(supplement, f, ensure_ascii=False)
    print(f'Saved: {OUT_PATH}')

    print('4. Building proper-noun (Np) augmentation table...', flush=True)
    # Group capitalized surface forms by their lowercase ASCII key.
    cap_dist = defaultdict(dict)
    for form, n in cap_freq.items():
        if n < 5:
            continue
        ascii_key = strip_diacritics(form).lower()
        cap_dist[ascii_key][form] = n

    augment = {}
    for ascii_key, variants in cap_dist.items():
        best, best_n = max(variants.items(), key=lambda kv: kv[1])
        if not (set(best.lower()) & DIAC_CHARS):
            continue  # capitalized ASCII variant dominates -> nothing to add
        identity = ascii_key[0].upper() + ascii_key[1:]
        identity_n = variants.get(identity, 0)
        if best_n < DOMINANCE * max(identity_n, 1):
            continue
        augment[ascii_key] = {'word': best, 'freq': best_n}

    with open(OUT_NP, 'w', encoding='utf-8') as f:
        json.dump(augment, f, ensure_ascii=False)
    print(f'   Np augmentation: {len(augment):,} entries')
    for probe in ['bus', 'sirak', 'kusner', 'damaska', 'esdaun', 'zoze', 'crna']:
        print(f'   probe {probe!r}: {augment.get(probe)}')
    print(f'Saved: {OUT_NP}')


if __name__ == '__main__':
    main()
