#!/usr/bin/env python3
"""
Statistical significance tests for per-POS (Table 2) and ablation (Table 7).

McNemar's test per POS category and per ablation configuration.

"""

import os
import gzip
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats

# Same constants/lemmatizer as before
EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
# Final correction table: 17 manual + 131 reviewed data-driven pairs (148).
import csv as _csv
with open(Path(__file__).parent.parent / 'data' / 'lemma_corrections_v2.csv',
          encoding='utf-8') as _f:
    LEMMA_CORRECTIONS = {r['source_lemma']: r['corrected_lemma']
                         for r in _csv.DictReader(_f)}


def normalize_pnpa(tag):
    """Pn*/Pa* -> P* (srLex and the supplement have no Pn/Pa MSDs)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag
_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


class Lemmatizer:
    def __init__(self, srlex_path, expanded_dict_path=None, use_pos=True, use_corrections=True,
                 use_participle=True, use_hteti=True, use_diacritics=True,
                 use_suffix=True, v6_restorer=None):
        # Layer 0 = puni v6 modul (Sekcija 4.3); deljen ili kreiran ovde
        if use_diacritics:
            if v6_restorer is not None:
                self._v6 = v6_restorer
            else:
                sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
                from v6_restorer import V6Restorer
                self._v6 = V6Restorer(srlex_path, extra_dict_path=expanded_dict_path)
        else:
            self._v6 = None
        self.use_pos = use_pos
        self.use_corrections = use_corrections
        self.use_participle = use_participle
        self.use_hteti = use_hteti
        self.use_diacritics = use_diacritics
        self.use_suffix = use_suffix
        self._msd_index = defaultdict(dict)
        self._pos_index = defaultdict(lambda: defaultdict(list))
        self._word_index = defaultdict(list)
        self._ascii_index = defaultdict(list)

        msd_d = defaultdict(lambda: defaultdict(lambda: (None, 0)))
        pos_d = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        word_d = defaultdict(lambda: defaultdict(int))

        with gzip.open(srlex_path, 'rt', encoding='utf-8') as f:
            for line in f:
                p = line.strip().split('\t')
                if len(p) < 7: continue
                w, l, m = p[0].lower(), p[1].lower(), p[2]
                if self.use_corrections:
                    l = EKAVIZATION.get(l, l)
                try: fr = int(p[6])
                except: fr = 0
                pos = m[0] if m else '?'
                ex = msd_d[w][m]
                if fr > ex[1]: msd_d[w][m] = (l, fr)
                pos_d[w][pos][l] += fr
                word_d[w][l] += fr

        # Load expanded dictionary (Step 5)
        if expanded_dict_path and os.path.exists(expanded_dict_path):
            with open(expanded_dict_path) as f:
                extra = json.load(f)
            for w, entries in extra.items():
                wl = w.lower()
                for entry in entries:
                    l, m = entry[0].lower(), entry[1]
                    fr = entry[2] if len(entry) > 2 else 1000
                    if self.use_corrections:
                        l = EKAVIZATION.get(l, l)
                    pos = m[0] if m else '?'
                    ex = msd_d[wl][m]
                    if fr > ex[1]: msd_d[wl][m] = (l, fr)
                    pos_d[wl][pos][l] += fr
                    word_d[wl][l] += fr

        if self.use_diacritics:
            ascii_freq = defaultdict(lambda: defaultdict(int))
            for w, lf in word_d.items():
                af = w.translate(_DIAC_TO_ASCII)
                ascii_freq[af][w] += sum(lf.values())
            for af, wf in ascii_freq.items():
                self._ascii_index[af] = sorted(wf.items(), key=lambda x: -x[1])

        for w, ms in msd_d.items():
            self._msd_index[w] = {m: l for m, (l, f) in ms.items()}
        for w, ps in pos_d.items():
            for pos, lf in ps.items():
                self._pos_index[w][pos] = sorted(lf.items(), key=lambda x: -x[1])
        for w, lf in word_d.items():
            self._word_index[w] = sorted(lf.items(), key=lambda x: -x[1])

    def _correct(self, lemma):
        if not lemma: return lemma
        if self.use_corrections:
            lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
            return EKAVIZATION.get(lemma, lemma)
        return lemma

    def lemmatize(self, word, msd=""):
        msd = normalize_pnpa(msd)
        w = word.lower()
        pos = msd[0] if msd else ""
        if self.use_hteti and w in _HTETI_FORMS and pos == 'V': return 'hteti'
        # Layer 0: puni modul za restauraciju dijakritika (Sekcija 4.3)
        if (self.use_diacritics and self._v6 is not None
                and w not in self._word_index and not (_DIAC_CHARS & set(word))):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma: return self._correct(lemma)
        if self.use_pos and pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            if self.use_participle and msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')): return self._correct(l)
            for l, f in cands:
                if l == w: return w
            return self._correct(cands[0][0])
        if msd and w in self._msd_index:
            for plen in range(len(msd)-1, 0, -1):
                for fm, l in self._msd_index[w].items():
                    if fm.startswith(msd[:plen]): return self._correct(l)
        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w: return w
            return self._correct(self._word_index[w][0][0])
        if self.use_suffix and pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if guess in self._word_index: return guess
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if guess in self._word_index: return guess
        if word[0].isupper() and len(word) > 1 and word[1:].islower(): return word
        return w


def load_conllu(path):
    sentences, current = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current: sentences.append(current); current = []
                continue
            if line.startswith('#'): continue
            p = line.split('\t')
            if len(p) < 6 or '-' in p[0] or '.' in p[0]: continue
            current.append((p[1], p[2], p[4]))
    if current: sentences.append(current)
    return sentences


def mcnemar_test(correct_a, correct_b):
    """McNemar's test between two binary arrays."""
    b = int(((correct_a == 1) & (correct_b == 0)).sum())  # only A correct
    c = int(((correct_a == 0) & (correct_b == 1)).sum())  # only B correct
    if (b + c) == 0:
        return 1.0, b, c
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p = 1 - stats.chi2.cdf(chi2, 1)
    return p, b, c


def main():
    base = Path(__file__).resolve().parent.parent.parent
    if not (base / 'POS-Aware-Stemmer').exists() and (base.parent / 'POS-Aware-Stemmer').exists():
        base = base.parent  # repo/scripts layout is one level deeper
    srlex = str(base / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    expanded = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')
    test_path = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'

    print("Loading test data...", flush=True)
    gold = load_conllu(test_path)
    n = sum(len(s) for s in gold)
    print(f"  {len(gold)} sentences, {n:,} tokens", flush=True)
    print(f"  Using expanded dictionary: {os.path.exists(expanded)}\n", flush=True)

    # ================================================================
    # PART 1: Per-POS McNemar (Table 2) - Our system vs CLASSLA
    # ================================================================
    print("=" * 60)
    print("PART 1: Per-POS McNemar tests (Our system vs CLASSLA)")
    print("=" * 60)

    # Load our lemmatizer (with expanded dict = final system)
    print("Loading lemmatizer (expanded dict)...", flush=True)
    lem = Lemmatizer(srlex, expanded_dict_path=expanded)

    # Get our predictions
    our_correct = []
    pos_labels = []
    for sent in gold:
        for w, gl, msd in sent:
            pred = lem.lemmatize(w, msd)
            our_correct.append(1 if (pred and gl and pred.lower() == gl.lower()) else 0)
            pos_labels.append(msd[0] if msd else '?')
    our_correct = np.array(our_correct)
    pos_labels = np.array(pos_labels)

    # Load CLASSLA predictions (from previous eval or re-run)
    # We need to load CLASSLA results. Let's check if we have them cached.
    classla_python = '/tmp/classla_env/bin/python3'
    classla_available = Path(classla_python).exists()

    if classla_available:
        import subprocess
        classla_script = '''
import classla
nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma",
                        tokenize_pretokenized=True, use_gpu=False)
sentences = []
current = []
with open("/tmp/UD_Serbian-SET/sr_set-ud-test.conllu") as f:
    for line in f:
        line = line.strip()
        if not line:
            if current: sentences.append(current); current = []
            continue
        if line.startswith("#"): continue
        parts = line.split("\\t")
        if len(parts) >= 2 and "-" not in parts[0] and "." not in parts[0]:
            current.append(parts[1])
if current: sentences.append(current)
doc = nlp([[w for w in sent] for sent in sentences])
for sent in doc.sentences:
    lemmas = [w.lemma for w in sent.words]
    print("\\t".join(lemmas))
'''
        tmp = '/tmp/classla_perpos_test.py'
        with open(tmp, 'w') as f:
            f.write(classla_script)
        print("Running CLASSLA on UD test...", flush=True)
        result = subprocess.run([classla_python, tmp],
                               capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            cl_lemmas = [line.split('\t') for line in result.stdout.strip().split('\n') if line.strip()]
            cl_correct = []
            idx = 0
            for i, sent in enumerate(gold):
                if i >= len(cl_lemmas): break
                for j, (w, gl, msd) in enumerate(sent):
                    if j < len(cl_lemmas[i]):
                        cl_pred = cl_lemmas[i][j]
                        cl_correct.append(1 if (cl_pred and gl and cl_pred.lower() == gl.lower()) else 0)
                    else:
                        cl_correct.append(0)
            cl_correct = np.array(cl_correct)

            # Per-POS McNemar
            print(f"\n{'POS':<6} {'N':>6} {'Ours':>7} {'CL':>7} {'Δ':>7} {'p':>8} {'sig':>4}")
            print("-" * 50)
            for pos in ['N', 'V', 'A', 'P', 'R', 'S', 'C', 'M', 'Q']:
                mask = pos_labels == pos
                if mask.sum() == 0: continue
                our_pos = our_correct[mask]
                cl_pos = cl_correct[mask]
                our_acc = 100 * our_pos.mean()
                cl_acc = 100 * cl_pos.mean()
                p_val, b, c = mcnemar_test(our_pos, cl_pos)
                sig = '*' if p_val < 0.05 else ''
                print(f"  {pos:<4} {mask.sum():>6} {our_acc:>6.2f}% {cl_acc:>6.2f}% {our_acc-cl_acc:>+6.2f} {p_val:>8.4f} {sig:>4}")
        else:
            print(f"CLASSLA error: {result.stderr[:200]}")

    # ================================================================
    # PART 2: Ablation McNemar (Table 7) - Each config vs Full system
    # ================================================================
    print(f"\n{'='*60}")
    print("PART 2: Ablation McNemar tests (each config vs full system)")
    print("=" * 60)

    configs = [
        ("Full system", {}),
        ("− diacritics", dict(use_diacritics=False)),
        ("− POS info", dict(use_pos=False)),
        ("− corrections", dict(use_corrections=False)),
        ("− participle", dict(use_participle=False)),
        ("− hteti", dict(use_hteti=False)),
        ("− suffix", dict(use_suffix=False)),
        ("Bare dictionary", dict(use_pos=False, use_corrections=False,
            use_participle=False, use_hteti=False, use_suffix=False,
            use_diacritics=False)),
    ]

    full_correct = our_correct  # already computed
    print(f"\n{'Config':<25} {'Acc':>7} {'Δ':>7} {'p':>10} {'sig':>4}")
    print("-" * 55)

    for name, kwargs in configs:
        if not kwargs:
            print(f"  {'Full system':<23} {100*full_correct.mean():>6.2f}%     ---        ---")
            continue
        lem_abl = Lemmatizer(srlex, expanded_dict_path=expanded, **kwargs)
        abl_correct = []
        for sent in gold:
            for w, gl, msd in sent:
                pred = lem_abl.lemmatize(w, msd)
                abl_correct.append(1 if (pred and gl and pred.lower() == gl.lower()) else 0)
        abl_correct = np.array(abl_correct)
        acc = 100 * abl_correct.mean()
        delta = acc - 100 * full_correct.mean()
        p_val, b, c = mcnemar_test(full_correct, abl_correct)
        sig = '*' if p_val < 0.05 else ''
        print(f"  {name:<23} {acc:>6.2f}% {delta:>+6.2f} {p_val:>10.4f} {sig:>4}")
        del lem_abl

    # Save results
    print(f"\n{'='*60}")
    print("DONE")


if __name__ == '__main__':
    main()
