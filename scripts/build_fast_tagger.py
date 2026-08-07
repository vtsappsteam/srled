#!/usr/bin/env python3
"""
Convert the NLTK perceptron tagger to the compact vectorized format
and verify that predictions are identical on the UD test split.

"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from fast_tagger import FastPerceptronTagger, build_from_nltk

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'NLTK Treniranje').exists() and (BASE.parent / 'NLTK Treniranje').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper
NLTK_MODEL = BASE / 'NLTK Treniranje' / 'models_v7_pnpa' / 'perceptron-tagger-srwac-pnpa.pickle'
OUT_DIR = BASE / 'NLTK Treniranje' / 'models_v8_fast_pnpa'
TEST_PATH = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'


def load_conllu(path):
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            p = line.split('\t')
            if len(p) < 6 or '-' in p[0] or '.' in p[0]:
                continue
            current.append(p[1])
    if current:
        sentences.append(current)
    return sentences


def main():
    sents = load_conllu(TEST_PATH)
    n_tokens = sum(len(s) for s in sents)
    print(f"Verification workload: {len(sents)} sentences, {n_tokens:,} tokens", flush=True)

    # Build compact model
    print("\nConverting NLTK model -> compact format...", flush=True)
    t0 = time.perf_counter()
    stats = build_from_nltk(NLTK_MODEL, OUT_DIR)
    print(f"  Done in {time.perf_counter()-t0:.0f}s: {stats['features']:,} features, "
          f"{stats['nnz']:,} weights, {stats['classes']} classes", flush=True)

    total_size = sum(f.stat().st_size for f in OUT_DIR.iterdir()) / 1e6
    print(f"  Model size on disk: {total_size:.0f} MB", flush=True)

    # Reference predictions from NLTK
    print("\nTagging with NLTK tagger (reference)...", flush=True)
    with open(NLTK_MODEL, 'rb') as f:
        nltk_tagger = pickle.load(f)
    t0 = time.perf_counter()
    ref = [nltk_tagger.tag(s) for s in sents]
    t_ref = time.perf_counter() - t0
    print(f"  {n_tokens/t_ref:,.0f} tok/s", flush=True)
    del nltk_tagger

    # Fast tagger predictions
    print("\nLoading fast tagger...", flush=True)
    t0 = time.perf_counter()
    fast = FastPerceptronTagger.load(OUT_DIR)
    print(f"  Loaded in {time.perf_counter()-t0:.2f}s", flush=True)

    t0 = time.perf_counter()
    hyp = [fast.tag(s) for s in sents]
    t_fast = time.perf_counter() - t0
    print(f"  {n_tokens/t_fast:,.0f} tok/s ({t_ref/t_fast:.1f}x faster)", flush=True)

    # Verify identity
    mism = 0
    for rs, hs in zip(ref, hyp):
        for (w1, t1), (w2, t2) in zip(rs, hs):
            if t1 != t2:
                mism += 1
                if mism <= 10:
                    print(f"  MISMATCH: {w1!r} nltk={t1} fast={t2}", flush=True)
    print(f"\nIdentity check: {n_tokens - mism}/{n_tokens} identical "
          f"({'PASS' if mism == 0 else 'FAIL: ' + str(mism) + ' mismatches'})", flush=True)


if __name__ == '__main__':
    main()
