#!/usr/bin/env python3
"""
Efficiency benchmark for the optimized pipeline
(vectorized memory-mapped tagger + compact memory-mapped lexicon).

Same methodology as benchmark_efficiency.py (warm-up, repeated runs,
mean ± std, peak RSS), plus a parallel batch scenario using fork-based
worker processes that share the memory-mapped model pages.

"""

import json
import multiprocessing as mp
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from fast_tagger import FastPerceptronTagger
from compact_lemmatizer import CompactLemmatizer

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'NLTK Treniranje').exists() and (BASE.parent / 'NLTK Treniranje').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper
TAGGER_DIR = BASE / 'NLTK Treniranje' / 'models_v8_fast_pnpa'
LEXICON_DIR = BASE / 'Diacritics-Restoration' / 'data' / 'compact_lexicon'
TEST_PATH = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT_PATH = Path(__file__).parent.parent / 'results' / 'benchmark_optimized.json'

REPS = 5
REPL_LEMMA = 20
N_WORKERS = 8

_TAGGER = None
_LEM = None


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
            current.append((p[1], p[4]))
    if current:
        sentences.append(current)
    return sentences


def peak_rss_gb(children=False):
    who = resource.RUSAGE_CHILDREN if children else resource.RUSAGE_SELF
    return resource.getrusage(who).ru_maxrss / (1024 ** 3)


def _process_chunk(sent_words):
    out = 0
    for words in sent_words:
        tagged = _TAGGER.tag(words)
        for w, m in tagged:
            _LEM.lemmatize(w, m)
        out += len(words)
    return out


def main():
    global _TAGGER, _LEM

    test = load_conllu(TEST_PATH)
    sent_words = [[w for w, m in s] for s in test]
    sent_msds = [[m for w, m in s] for s in test]
    n_tokens = sum(len(s) for s in test)
    n_sents = len(test)
    print(f"Workload: {n_sents} sentences, {n_tokens:,} tokens", flush=True)

    # Load (memory-mapped)
    t0 = time.perf_counter()
    _TAGGER = FastPerceptronTagger.load(TAGGER_DIR)
    load_tag = time.perf_counter() - t0
    t0 = time.perf_counter()
    _LEM = CompactLemmatizer(LEXICON_DIR)
    load_lem = time.perf_counter() - t0
    rss_load = peak_rss_gb()
    print(f"Load: tagger {load_tag:.2f}s, lexicon {load_lem:.2f}s, RSS {rss_load:.2f} GB", flush=True)

    # Warm-up (touches mmap pages)
    for words in sent_words:
        tagged = _TAGGER.tag(words)
        for w, m in tagged:
            _LEM.lemmatize(w, m)

    # (a) Lemmatization-only (compact backend), gold MSD
    times_a = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for _ in range(REPL_LEMMA):
            for words, msds in zip(sent_words, sent_msds):
                for w, m in zip(words, msds):
                    _LEM.lemmatize(w, m)
        times_a.append(time.perf_counter() - t0)
    tput_a = [n_tokens * REPL_LEMMA / t for t in times_a]
    print(f"\n(a) Lemmatization-only (compact): "
          f"{statistics.mean(tput_a):,.0f} ± {statistics.stdev(tput_a):,.0f} tok/s", flush=True)

    # (b) End-to-end single-stream (one sentence at a time, single core)
    times_b = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        for words in sent_words:
            tagged = _TAGGER.tag(words)
            for w, m in tagged:
                _LEM.lemmatize(w, m)
        times_b.append(time.perf_counter() - t0)
    tput_b = [n_tokens / t for t in times_b]
    lat_b = [t / n_sents * 1000 for t in times_b]
    print(f"(b) End-to-end single-stream: "
          f"{statistics.mean(tput_b):,.0f} ± {statistics.stdev(tput_b):,.0f} tok/s, "
          f"{statistics.mean(lat_b):.2f} ± {statistics.stdev(lat_b):.2f} ms/sentence", flush=True)

    # (c) End-to-end parallel batch (fork workers share mmap pages)
    ctx = mp.get_context('fork')
    chunks = [sent_words[i::N_WORKERS] for i in range(N_WORKERS)]
    times_c = []
    with ctx.Pool(N_WORKERS) as pool:
        pool.map(_process_chunk, [c[:5] for c in chunks])  # pool warm-up
        for _ in range(REPS):
            t0 = time.perf_counter()
            pool.map(_process_chunk, chunks)
            times_c.append(time.perf_counter() - t0)
    tput_c = [n_tokens / t for t in times_c]
    print(f"(c) End-to-end parallel batch ({N_WORKERS} workers): "
          f"{statistics.mean(tput_c):,.0f} ± {statistics.stdev(tput_c):,.0f} tok/s", flush=True)

    rss_self = peak_rss_gb()
    rss_child = peak_rss_gb(children=True)
    print(f"\nPeak RSS: parent {rss_self:.2f} GB, max child {rss_child:.2f} GB", flush=True)

    results = {
        'workload': {'sentences': n_sents, 'tokens': n_tokens, 'source': 'UD Serbian-SET test'},
        'methodology': {'reps': REPS, 'warmup': True, 'timer': 'time.perf_counter',
                        'memory': 'ru_maxrss (peak RSS)',
                        'hardware': 'Apple M2, 16 GB RAM', 'workers': N_WORKERS},
        'load_time_s': {'tagger': round(load_tag, 2), 'lexicon': round(load_lem, 2)},
        'lemma_only_compact': {
            'throughput_mean': round(statistics.mean(tput_a)),
            'throughput_std': round(statistics.stdev(tput_a)),
        },
        'end_to_end_single': {
            'throughput_mean': round(statistics.mean(tput_b)),
            'throughput_std': round(statistics.stdev(tput_b)),
            'latency_ms_mean': round(statistics.mean(lat_b), 2),
            'latency_ms_std': round(statistics.stdev(lat_b), 2),
        },
        'end_to_end_parallel': {
            'throughput_mean': round(statistics.mean(tput_c)),
            'throughput_std': round(statistics.stdev(tput_c)),
            'workers': N_WORKERS,
        },
        'peak_rss_gb': {'parent': round(rss_self, 2), 'max_child': round(rss_child, 2)},
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
