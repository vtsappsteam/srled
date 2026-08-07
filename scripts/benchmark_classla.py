#!/usr/bin/env python3
"""
Rigorous efficiency benchmark for CLASSLA-Stanza (run inside /tmp/classla_env).

Same methodology as benchmark_efficiency.py: load excluded from timing,
one warm-up pass, repeated timed runs (3, given ~25 s per run), mean and
standard deviation, peak RSS via resource.getrusage.

Run: /tmp/classla_env/bin/python3 benchmark_classla.py

"""

import json
import resource
import statistics
import time
from pathlib import Path

TEST_PATH = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT_PATH = Path(__file__).parent.parent / 'results' / 'benchmark_classla.json'
REPS = 3


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


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
    import classla

    sents = load_conllu(TEST_PATH)
    n_tokens = sum(len(s) for s in sents)
    n_sents = len(sents)
    print(f"Workload: {n_sents} sentences, {n_tokens:,} tokens", flush=True)

    t0 = time.perf_counter()
    nlp = classla.Pipeline('sr', type='standard', processors='tokenize,pos,lemma',
                           tokenize_pretokenized=True, use_gpu=False)
    load_t = time.perf_counter() - t0
    print(f"Pipeline load: {load_t:.1f}s", flush=True)

    # Warm-up
    nlp([s for s in sents[:50]])

    times = []
    for r in range(REPS):
        t0 = time.perf_counter()
        nlp(sents)
        times.append(time.perf_counter() - t0)
        print(f"  run {r+1}: {times[-1]:.1f}s", flush=True)

    tput = [n_tokens / t for t in times]
    lat = [t / n_sents * 1000 for t in times]
    peak = peak_rss_gb()

    print(f"\nThroughput: {statistics.mean(tput):,.0f} ± {statistics.stdev(tput):,.0f} tok/s", flush=True)
    print(f"Latency: {statistics.mean(lat):.1f} ± {statistics.stdev(lat):.1f} ms/sentence", flush=True)
    print(f"Peak RSS: {peak:.2f} GB", flush=True)

    results = {
        'workload': {'sentences': n_sents, 'tokens': n_tokens, 'source': 'UD Serbian-SET test'},
        'methodology': {'reps': REPS, 'warmup': True, 'timer': 'time.perf_counter',
                        'memory': 'ru_maxrss (peak RSS)', 'use_gpu': False,
                        'processors': 'tokenize,pos,lemma', 'pretokenized': True},
        'load_time_s': round(load_t, 1),
        'throughput_mean': round(statistics.mean(tput)),
        'throughput_std': round(statistics.stdev(tput)),
        'latency_ms_per_sent_mean': round(statistics.mean(lat), 1),
        'latency_ms_per_sent_std': round(statistics.stdev(lat), 1),
        'times_s': [round(t, 1) for t in times],
        'peak_rss_gb': round(peak, 2),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
