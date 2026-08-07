#!/usr/bin/env python3
"""
Rigorous efficiency benchmark for the proposed pipeline.

Methodology (following Peng et al. 2023 "Efficiency Pentathlon" and
Treviso et al. 2023 TACL survey):
  - Model loading time reported separately; measurement starts after load
  - One warm-up pass before timed runs
  - 5 timed repetitions, mean and standard deviation reported
  - Workload replicated so each timed window is long enough to be stable
  - Peak memory = maximum resident set size (resource.getrusage, stdlib)
  - Two configurations benchmarked:
      (a) lemmatization-only, MSD tags precomputed (dictionary lookup speed)
      (b) end-to-end: POS tagging + lemmatization (fair vs CLASSLA-Stanza)

"""

import json
import pickle
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ablation_study import AblationLemmatizer, load_conllu

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'POS-Aware-Stemmer').exists() and (BASE.parent / 'POS-Aware-Stemmer').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper
SRLEX = str(BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
EXPANDED = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')
V7_MODEL = str(BASE / 'NLTK Treniranje' / 'models_v7_pnpa' / 'perceptron-tagger-srwac-pnpa.pickle')
TEST_PATH = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT_PATH = Path(__file__).parent.parent / 'results' / 'benchmark_efficiency.json'

REPS = 5          # timed repetitions
REPL_LEMMA = 20   # corpus replications for lemma-only (tiny per-pass time)


def peak_rss_gb():
    # macOS reports ru_maxrss in bytes
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


def main():
    test = load_conllu(TEST_PATH)
    sent_words = [[w for w, l, m in s] for s in test]
    sent_msds = [[m for w, l, m in s] for s in test]
    n_tokens = sum(len(s) for s in test)
    n_sents = len(test)
    print(f"Workload: {n_sents} sentences, {n_tokens:,} tokens (UD test split)", flush=True)

    # ---- Load models (reported separately, excluded from throughput) ----
    t0 = time.perf_counter()
    lem = AblationLemmatizer(SRLEX, expanded_dict_path=EXPANDED)
    load_lem = time.perf_counter() - t0
    rss_lem = peak_rss_gb()
    print(f"Lemmatizer load: {load_lem:.1f}s, RSS: {rss_lem:.2f} GB", flush=True)

    t0 = time.perf_counter()
    with open(V7_MODEL, 'rb') as f:
        tagger = pickle.load(f)
    load_tag = time.perf_counter() - t0
    rss_full = peak_rss_gb()
    print(f"Tagger load: {load_tag:.1f}s, RSS after both: {rss_full:.2f} GB", flush=True)

    # ---- (a) Lemmatization-only, precomputed MSD ----
    # Warm-up
    for words, msds in zip(sent_words, sent_msds):
        for w, m in zip(words, msds):
            lem.lemmatize(w, m)

    times_a = []
    for r in range(REPS):
        t0 = time.perf_counter()
        for _ in range(REPL_LEMMA):
            for words, msds in zip(sent_words, sent_msds):
                for w, m in zip(words, msds):
                    lem.lemmatize(w, m)
        times_a.append(time.perf_counter() - t0)
    tput_a = [n_tokens * REPL_LEMMA / t for t in times_a]
    print(f"\n(a) Lemmatization-only (precomputed MSD), {REPS} runs x {REPL_LEMMA} replications:", flush=True)
    print(f"    Throughput: {statistics.mean(tput_a):,.0f} ± {statistics.stdev(tput_a):,.0f} tok/s", flush=True)

    # ---- (b) End-to-end: POS tagging + lemmatization ----
    # Warm-up
    for words in sent_words[:100]:
        tagged = tagger.tag(words)
        for w, m in tagged:
            lem.lemmatize(w, m)

    times_b = []
    for r in range(REPS):
        t0 = time.perf_counter()
        for words in sent_words:
            tagged = tagger.tag(words)
            for w, m in tagged:
                lem.lemmatize(w, m)
        times_b.append(time.perf_counter() - t0)
    tput_b = [n_tokens / t for t in times_b]
    lat_b = [t / n_sents * 1000 for t in times_b]  # ms per sentence
    print(f"\n(b) End-to-end (tagging + lemmatization), {REPS} runs:", flush=True)
    print(f"    Throughput: {statistics.mean(tput_b):,.0f} ± {statistics.stdev(tput_b):,.0f} tok/s", flush=True)
    print(f"    Latency: {statistics.mean(lat_b):.2f} ± {statistics.stdev(lat_b):.2f} ms/sentence", flush=True)

    peak = peak_rss_gb()
    print(f"\nPeak RSS (whole process): {peak:.2f} GB", flush=True)

    results = {
        'workload': {'sentences': n_sents, 'tokens': n_tokens, 'source': 'UD Serbian-SET test'},
        'methodology': {'reps': REPS, 'warmup': True, 'timer': 'time.perf_counter',
                        'memory': 'ru_maxrss (peak RSS)', 'hardware': 'Apple M2, 16 GB RAM, single core'},
        'load_time_s': {'lemmatizer': round(load_lem, 2), 'tagger': round(load_tag, 2)},
        'lemma_only': {
            'throughput_mean': round(statistics.mean(tput_a)),
            'throughput_std': round(statistics.stdev(tput_a)),
            'times_s': [round(t, 3) for t in times_a],
            'replications': REPL_LEMMA,
        },
        'end_to_end': {
            'throughput_mean': round(statistics.mean(tput_b)),
            'throughput_std': round(statistics.stdev(tput_b)),
            'latency_ms_per_sent_mean': round(statistics.mean(lat_b), 2),
            'latency_ms_per_sent_std': round(statistics.stdev(lat_b), 2),
            'times_s': [round(t, 3) for t in times_b],
        },
        'peak_rss_gb': round(peak, 2),
        'rss_after_lemmatizer_gb': round(rss_lem, 2),
        'rss_after_both_gb': round(rss_full, 2),
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_PATH}", flush=True)


if __name__ == '__main__':
    main()
