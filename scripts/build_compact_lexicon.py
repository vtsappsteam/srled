#!/usr/bin/env python3
"""
Build the compact memory-mapped lexicon from srLex + expanded dictionary
and verify that predictions are identical to the in-memory lemmatizer
on the UD test and dev splits.

"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from ablation_study import AblationLemmatizer, LEMMA_CORRECTIONS, load_conllu
from compact_lemmatizer import CompactLemmatizer, build_from_lemmatizer

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'POS-Aware-Stemmer').exists() and (BASE.parent / 'POS-Aware-Stemmer').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper
SRLEX = str(BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
EXPANDED = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')
OUT_DIR = BASE / 'Diacritics-Restoration' / 'data' / 'compact_lexicon'


def main():
    test = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    dev = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-dev.conllu')
    sents = test + dev
    n_tokens = sum(len(s) for s in sents)
    print(f"Verification workload: {len(sents)} sentences, {n_tokens:,} tokens (test+dev)", flush=True)

    print("\nLoading in-memory lemmatizer (reference)...", flush=True)
    t0 = time.perf_counter()
    ref = AblationLemmatizer(SRLEX, expanded_dict_path=EXPANDED)
    print(f"  Loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    print("Building compact lexicon...", flush=True)
    t0 = time.perf_counter()
    stats = build_from_lemmatizer(ref, OUT_DIR, lemma_corrections=LEMMA_CORRECTIONS)
    print(f"  Built in {time.perf_counter()-t0:.0f}s: {stats['words']:,} words, "
          f"{stats['lemmas']:,} lemmas, {stats['msds']} MSDs", flush=True)

    print("Building compact v6 diacritics tables...", flush=True)
    from compact_v6 import build_v6_tables
    t0 = time.perf_counter()
    v6stats = build_v6_tables(ref._v6.cg, OUT_DIR)
    print(f"  Built in {time.perf_counter()-t0:.0f}s: "
          f"{v6stats['ascii_keys']:,} ascii keys, {v6stats['words']:,} words, "
          f"{v6stats['msds']} MSDs", flush=True)
    size = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / 1e6
    print(f"  Size on disk: {size:.0f} MB", flush=True)

    print("\nLoading compact lemmatizer...", flush=True)
    t0 = time.perf_counter()
    compact = CompactLemmatizer(OUT_DIR)
    print(f"  Loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    print("\nVerifying identity on gold MSD tags...", flush=True)
    mism = 0
    checked = 0
    t_ref = t_cmp = 0.0
    for sent in sents:
        for w, gl, msd in sent:
            t0 = time.perf_counter()
            a = ref.lemmatize(w, msd)
            t_ref += time.perf_counter() - t0
            t0 = time.perf_counter()
            b = compact.lemmatize(w, msd)
            t_cmp += time.perf_counter() - t0
            checked += 1
            if a != b:
                mism += 1
                if mism <= 10:
                    print(f"  MISMATCH: {w!r} msd={msd} ref={a!r} compact={b!r}", flush=True)

    print(f"\nIdentity: {checked - mism}/{checked} "
          f"({'PASS' if mism == 0 else 'FAIL: ' + str(mism)})", flush=True)
    print(f"Reference:  {checked/t_ref:,.0f} tok/s", flush=True)
    print(f"Compact:    {checked/t_cmp:,.0f} tok/s", flush=True)


if __name__ == '__main__':
    main()
