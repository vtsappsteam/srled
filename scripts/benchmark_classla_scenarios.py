#!/usr/bin/env python3
"""
Scenario-based CLASSLA-Stanza benchmark for Table 12 (run under
/tmp/classla_env/bin/python3):
  - load time (separately reported)
  - warm-up pass before all timed runs
  - batch: whole corpus in one call, 3 reps, mean +- std
  - single-stream: one sentence per call, 3 reps over the corpus
  - peak RSS via resource.getrusage

"""
import json
import resource
import time
from pathlib import Path

BASE = Path(__file__).parent.parent
TEST_PATH = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'
OUT = str(BASE / 'results' / 'benchmark_classla_scenarios.json')


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
    print(f'Workload: {len(sents)} sentences, {n_tokens:,} tokens', flush=True)

    t0 = time.perf_counter()
    nlp = classla.Pipeline('sr', type='standard',
                           processors='tokenize,pos,lemma',
                           tokenize_pretokenized=True, use_gpu=False)
    load_s = time.perf_counter() - t0
    print(f'Load: {load_s:.1f}s', flush=True)

    # warm-up: full corpus once (JIT, caches, lazy init)
    nlp(sents)
    print('Warm-up done', flush=True)

    batch_times = []
    for i in range(3):
        t0 = time.perf_counter()
        nlp(sents)
        batch_times.append(time.perf_counter() - t0)
        print(f'  batch run {i+1}: {batch_times[-1]:.2f}s', flush=True)

    ss_times = []
    for i in range(3):
        t0 = time.perf_counter()
        for s in sents:
            nlp([s])
        ss_times.append(time.perf_counter() - t0)
        print(f'  single-stream run {i+1}: {ss_times[-1]:.1f}s', flush=True)

    def ms(x):
        return [t / len(sents) * 1000 for t in x]

    def stat(v):
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5
        return m, sd

    bt_m, bt_sd = stat([n_tokens / t for t in batch_times])
    ss_tok_m, ss_tok_sd = stat([n_tokens / t for t in ss_times])
    ss_ms_m, ss_ms_sd = stat(ms(ss_times))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9

    res = {
        'workload': {'sentences': len(sents), 'tokens': n_tokens},
        'load_time_s': round(load_s, 1),
        'batch_tok_s': [round(bt_m), round(bt_sd)],
        'single_stream_tok_s': [round(ss_tok_m), round(ss_tok_sd)],
        'single_stream_ms_per_sent': [round(ss_ms_m, 1), round(ss_ms_sd, 1)],
        'peak_rss_gb': round(peak, 2),
    }
    print(json.dumps(res, indent=1))
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
