#!/usr/bin/env python3
"""
Stripped-ReLDI comparison restricted to the ReLDI TEST split (9,088 tokens):
the locked system's resources (supplement v2, data-driven corrections) are
mined from train splits only, so the test split is the leakage-free way to
report Table 6 with the full locked configuration.

Conditions (gold MSD, identical protocol to experiment_stripped_v6_full.py):
  B) locked lemmatizer alone on stripped test tokens
  C) Stage 3 restoration of every token, then locked lemmatizer
  D) CLASSLA end-to-end on stripped test tokens (rows extracted from the
     fresh full-corpus run stored in /tmp/classla_stripped_out.txt)

Also reports the same three conditions for the LEGACY srLex-only
configuration on the test split, for the protocol comparison.

Output: results/stripped_reldi_test_split.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'scripts'))
sys.path.insert(0, str(BASE / 'src'))

import evaluate_three_testsets as ev3
from experiment_stripped_reldi import (Lemmatizer as LegacyLemmatizer,
                                       strip_word)
from candidate_generator import CandidateGenerator
from pos_disambiguator import POSDisambiguator

RELDI = str(ev3.BASE / 'data' / 'ReLDI-NormTagNER-sr'
            / 'reldi-normtagner-sr.conllup')
CL_OUT = '/tmp/classla_stripped_out.txt'
OUT = str(BASE / 'results' / 'stripped_reldi_test_split.json')


def load_reldi_with_split(path):
    """Whole corpus in order, each sentence tagged with its split flag."""
    out, current, split = [], [], 'other'
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    out.append((split, current))
                current = []
                continue
            if line.startswith('# contained_in_datasets'):
                split = ('test' if 'test' in line
                         else 'train' if 'train' in line else 'other')
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            # identical token filter to experiment_stripped_reldi.load_reldi
            # (no lemma filter: '_' lemmas are kept)
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                current.append((parts[1], parts[2], parts[4]))
    if current:
        out.append((split, current))
    return out


def mcnemar_exact(a, b):
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    if bb + cc == 0:
        return 1.0, bb, cc
    return (round(float(stats.binomtest(min(bb, cc), bb + cc, 0.5).pvalue), 6),
            bb, cc)


def main():
    corpus = load_reldi_with_split(RELDI)
    cl_lines = [line.rstrip('\n').split('\t') for line in open(CL_OUT)]
    assert len(cl_lines) == len(corpus), \
        f'CLASSLA output mismatch: {len(cl_lines)} vs {len(corpus)}'

    test_idx = [i for i, (split, s) in enumerate(corpus) if split == 'test']
    n_tok = sum(len(corpus[i][1]) for i in test_idx)
    print(f'ReLDI test split: {len(test_idx)} sentences, {n_tok:,} tokens',
          flush=True)

    cg = CandidateGenerator(
        ev3.SRLEX,
        supplement_path=str(BASE / 'data' / 'srwac_supplement_v2.json'),
        augment_np_path=str(BASE / 'data' / 'srwac_augment_np.json'))
    disamb = POSDisambiguator(cg, tagger=None)

    res = {'tokens': n_tok, 'sentences': len(test_idx)}
    for cfg_name, make in (
            ('locked', lambda: ev3.Lemmatizer(ev3.SRLEX, ev3.EXTRA)),
            ('legacy_srlex_only', lambda: LegacyLemmatizer(ev3.SRLEX))):
        lem = make()
        vb, vc, vd = [], [], []
        for i in test_idx:
            sent = corpus[i][1]
            cl_sent = cl_lines[i]
            assert len(cl_sent) == len(sent), f'token mismatch sent {i}'
            for (w, gl, msd), cl_lemma in zip(sent, cl_sent):
                ws = strip_word(w)
                p = lem.lemmatize(ws, msd)
                vb.append(1 if p and gl and p.lower() == gl.lower() else 0)
                restored = disamb._restore_word_pos(ws, msd)
                p = lem.lemmatize(restored, msd)
                vc.append(1 if p and gl and p.lower() == gl.lower() else 0)
                vd.append(1 if gl and cl_lemma.lower() == gl.lower() else 0)
        del lem
        vb, vc, vd = np.array(vb), np.array(vc), np.array(vd)
        entry = {'B_lemmatizer_alone': round(100 * float(vb.mean()), 2),
                 'C_stage3_restoration': round(100 * float(vc.mean()), 2),
                 'D_classla': round(100 * float(vd.mean()), 2)}
        for pname, (x, y) in {'B_vs_D': (vb, vd), 'C_vs_D': (vc, vd),
                              'B_vs_C': (vb, vc)}.items():
            p, f_only, s_only = mcnemar_exact(x, y)
            entry[f'mcnemar_{pname}'] = {'p': p, 'first_only': f_only,
                                         'second_only': s_only}
        res[cfg_name] = entry
        print(cfg_name, json.dumps(entry), flush=True)

    with open(OUT, 'w') as f:
        json.dump(res, f, indent=1)
    print(f'Saved: {OUT}', flush=True)


if __name__ == '__main__':
    main()
