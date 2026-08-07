#!/usr/bin/env python3
"""
Full stripped-ReLDI comparison for Table 6 with the FINAL locked system
(srLex + expanded_supplement_v2 + 148 lemma corrections):
  B) proposed system, lemmatizer alone (its Layer 0 touches only OOV words)
  C) proposed system, Stage 3 restoration of every token, then lemmatizer
  D) CLASSLA-Stanza end-to-end on stripped text
with exact McNemar tests between all pairs (per-token correctness vectors).

Gold MSD protocol for B and C, identical to experiment_stripped_reldi.py.

SANITY: the previously published protocol (srLex-only lemmatizer with the
17 manual corrections, from experiment_stripped_reldi.py) is re-run first
and must reproduce the stored values (B 90.29 / C 93.14) with the fresh
CLASSLA run matching 93.90.

Usage (long-running, detached):
    nohup python3 scripts/experiment_stripped_v6_full.py \
        > results/stripped_reldi_v6_full.log 2>&1 &
Output: results/stripped_reldi_v6_full.json
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))

import evaluate_three_testsets as ev3
from experiment_stripped_reldi import (Lemmatizer as LegacyLemmatizer,
                                       load_reldi, strip_word)
from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator

ROOT = ev3.BASE
SRLEX = ev3.SRLEX
RELDI = str(ROOT / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup')
CLASSLA_PY = '/tmp/classla_env/bin/python3'
OUT = str(BASE / 'results' / 'stripped_reldi_v6_full.json')
LEGACY_REF = {'B': 90.29, 'C': 93.14, 'D': 93.9}


def mcnemar_exact(a, b):
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    if (bb + cc) == 0:
        return 1.0, bb, cc
    p = stats.binomtest(min(bb, cc), bb + cc, 0.5).pvalue
    return round(float(p), 6), bb, cc


def run_classla_stripped(reldi):
    tmp_input = '/tmp/reldi_stripped_tokens.txt'
    with open(tmp_input, 'w') as f:
        for sent in reldi:
            f.write('\n'.join(strip_word(w) for w, _, _ in sent) + '\n\n')
    script = f'''
import classla
nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma",
                       tokenize_pretokenized=True, use_gpu=False)
sentences = []
current = []
with open("{tmp_input}") as f:
    for line in f:
        line = line.rstrip("\\n")
        if not line:
            if current: sentences.append(current); current = []
            continue
        current.append(line)
if current: sentences.append(current)
doc = nlp(sentences)
with open("/tmp/classla_stripped_out.txt", "w") as f:
    for sent in doc.sentences:
        f.write("\\t".join(w.lemma if w.lemma else "_" for w in sent.words) + "\\n")
print("DONE")
'''
    with open('/tmp/classla_stripped_run.py', 'w') as f:
        f.write(script)
    r = subprocess.run([CLASSLA_PY, '/tmp/classla_stripped_run.py'],
                       capture_output=True, text=True, timeout=3600)
    if 'DONE' not in r.stdout:
        print('CLASSLA STDERR:', r.stderr[-500:])
        raise RuntimeError('CLASSLA failed')
    lemmas = [line.rstrip('\n').split('\t')
              for line in open('/tmp/classla_stripped_out.txt')]
    return lemmas


def eval_conditions(lem, disamb, reldi):
    """Return (vB, vC) correctness vectors for one lemmatizer config."""
    vb, vc = [], []
    for sent in reldi:
        for w, gl, msd in sent:
            ws = strip_word(w)
            p = lem.lemmatize(ws, msd)
            vb.append(1 if p and gl and p.lower() == gl.lower() else 0)
            restored = disamb._restore_word_pos(ws, msd)
            p = lem.lemmatize(restored, msd)
            vc.append(1 if p and gl and p.lower() == gl.lower() else 0)
    return np.array(vb), np.array(vc)


def main():
    print('1. Restoration module (Stage 3) tables...', flush=True)
    cg = CandidateGenerator(
        SRLEX,
        supplement_path=str(BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(BASE / 'data' / 'srwac_augment_np.json'))
    disamb = POSDisambiguator(cg, tagger=None)

    reldi = load_reldi(RELDI)
    total = sum(len(s) for s in reldi)
    print(f'   {len(reldi)} sentences, {total:,} tokens', flush=True)

    print('2. LEGACY protocol (srLex-only + 17 corrections) - sanity...',
          flush=True)
    legacy = LegacyLemmatizer(SRLEX)
    vb_leg, vc_leg = eval_conditions(legacy, disamb, reldi)
    del legacy
    b_leg = round(100 * float(vb_leg.mean()), 2)
    c_leg = round(100 * float(vc_leg.mean()), 2)
    print(f'   B_legacy {b_leg} (ref {LEGACY_REF["B"]}) | '
          f'C_legacy {c_leg} (ref {LEGACY_REF["C"]})', flush=True)
    sanity = (b_leg == LEGACY_REF['B'] and c_leg == LEGACY_REF['C'])

    print('3. LOCKED system (supplement v2 + 148 corrections)...', flush=True)
    lem = ev3.Lemmatizer(SRLEX, ev3.EXTRA)
    vb, vc = eval_conditions(lem, disamb, reldi)
    del lem
    print(f'   B {round(100 * float(vb.mean()), 2)} | '
          f'C {round(100 * float(vc.mean()), 2)}', flush=True)

    print('4. CLASSLA on stripped text (this takes a while)...', flush=True)
    cl = run_classla_stripped(reldi)
    assert len(cl) == len(reldi), f'sentence mismatch {len(cl)} vs {len(reldi)}'
    vd = []
    for si, sent in enumerate(reldi):
        cl_sent = cl[si]
        assert len(cl_sent) == len(sent), f'token mismatch in sent {si}'
        for (w, gl, msd), cl_lemma in zip(sent, cl_sent):
            vd.append(1 if gl and cl_lemma.lower() == gl.lower() else 0)
    vd = np.array(vd)
    d_acc = round(100 * float(vd.mean()), 2)
    sanity = sanity and (d_acc == LEGACY_REF['D'])
    print(f'   D {d_acc} (ref {LEGACY_REF["D"]})', flush=True)

    res = {
        'tokens': total,
        'config': ('locked: srLex + expanded_supplement_v2 + 148 corrections '
                   '+ Pn/Pa->P; Stage 3 = srwac_supplement_v2 + Np tables'),
        'B_lemmatizer_alone': round(100 * float(vb.mean()), 2),
        'C_stage3_restoration': round(100 * float(vc.mean()), 2),
        'D_classla': d_acc,
        'legacy_protocol_sanity': {
            'B': b_leg, 'C': c_leg, 'D': d_acc,
            'reference': LEGACY_REF,
            'pass': bool(sanity)},
    }
    for name, (x, y) in {'B_vs_D': (vb, vd), 'C_vs_D': (vc, vd),
                         'B_vs_C': (vb, vc)}.items():
        p, b_only, c_only = mcnemar_exact(x, y)
        res[f'mcnemar_{name}'] = {'p': p, 'first_only': b_only,
                                  'second_only': c_only}
    print(json.dumps(res, indent=1))
    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'Saved: {OUT}', flush=True)
    print(f'LEGACY SANITY: {"PASS" if sanity else "FAIL"}', flush=True)
    return 0 if sanity else 1


if __name__ == '__main__':
    sys.exit(main())
