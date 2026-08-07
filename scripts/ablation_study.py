#!/usr/bin/env python3
"""
Ablation study: contribution of each layer in the cascade lemmatizer.

Tests removing one component at a time to measure its contribution.

Configurations:
  1. Full system (all layers)
  2. No diacritics restoration (Layer 0 off)
  3. No POS information (frequency-only from dict)
  4. No morphological guesser (Layer 3 off)
  5. No lemma corrections (EKAV/CORR off)
  6. No participle rule
  7. No hteti special rule
  8. Bare dictionary only (no rules, no diacritics, no POS)

"""

import csv
import gzip
import json
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter

# Same constants as main system
EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
# Final correction table: 17 manual + 131 reviewed data-driven pairs (148).
_CORRECTIONS_CSV = Path(__file__).parent.parent / 'data' / 'lemma_corrections_v2.csv'
with open(_CORRECTIONS_CSV, encoding='utf-8') as _f:
    LEMMA_CORRECTIONS = {r['source_lemma']: r['corrected_lemma']
                         for r in csv.DictReader(_f)}


def normalize_pnpa(tag):
    """Pn*/Pa* -> P* (srLex and the supplement have no Pn/Pa MSDs)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


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


class AblationLemmatizer:
    """Configurable lemmatizer for ablation study."""

    def __init__(self, srlex_path,
                 expanded_dict_path=None,
                 use_diacritics=True,
                 use_pos=True,
                 use_corrections=True,
                 use_participle_rule=True,
                 use_hteti_rule=True,
                 use_morph_guesser=True,
                 use_expanded=True,
                 v6_restorer=None):
        self.use_diacritics = use_diacritics
        self.use_pos = use_pos
        self.use_corrections = use_corrections
        self.use_participle_rule = use_participle_rule
        self.use_hteti_rule = use_hteti_rule
        self.use_morph_guesser = use_morph_guesser

        self._msd_index = defaultdict(dict)
        self._pos_index = defaultdict(lambda: defaultdict(list))
        self._word_index = defaultdict(list)
        self._ascii_index = defaultdict(list)
        self._load(srlex_path, expanded_dict_path if use_expanded else None)

        # Layer 0 = puni modul za restauraciju dijakritika (Sekcija 4.3 rada).
        # Deljena instanca se prosleđuje (v6_restorer); ako nije data,
        # pravi se ovde sa istim rečničkim resursima kao lematizator.
        self._v6 = None
        if use_diacritics:
            if v6_restorer is not None:
                self._v6 = v6_restorer
            else:
                _src = str(Path(__file__).parent.parent / 'src')
                if _src not in sys.path:
                    sys.path.insert(0, _src)
                from v6_restorer import V6Restorer
                self._v6 = V6Restorer(
                    srlex_path,
                    extra_dict_path=expanded_dict_path if use_expanded else None)

    def _load(self, srlex_path, expanded_dict_path=None):
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
        if expanded_dict_path and Path(expanded_dict_path).exists():
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
        if not self.use_corrections or not lemma:
            return lemma
        lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        msd = normalize_pnpa(msd)
        w = word.lower()
        pos = msd[0] if msd else ""

        # Hteti rule
        if self.use_hteti_rule and w in _HTETI_FORMS and pos == 'V':
            return 'hteti'

        # Layer 0: restauracija dijakritika punim modulom (Sekcija 4.3)
        if (self.use_diacritics and self._v6 is not None
                and w not in self._word_index and not (_DIAC_CHARS & set(word))):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()

        # MSD exact
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma:
                return self._correct(lemma)

        # POS lookup
        if self.use_pos and pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            if self.use_participle_rule and msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')):
                        return self._correct(l)
            for l, f in cands:
                if l == w: return w
            return self._correct(cands[0][0])

        # MSD prefix
        if msd and w in self._msd_index:
            for plen in range(len(msd)-1, 0, -1):
                for fm, l in self._msd_index[w].items():
                    if fm.startswith(msd[:plen]):
                        return self._correct(l)

        # Freq
        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w: return w
            return self._correct(self._word_index[w][0][0])

        # Morph guesser
        if self.use_morph_guesser and pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if guess in self._word_index: return guess
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if guess in self._word_index: return guess

        # Identity
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word
        return w


def evaluate(lem, gold):
    """Return (accuracy, per-token 0/1 correctness list)."""
    vec = []
    for sent in gold:
        for w, gl, msd in sent:
            pred = lem.lemmatize(w, msd)
            vec.append(1 if pred and gl and pred.lower() == gl.lower() else 0)
    acc = round(100 * sum(vec) / len(vec), 2) if vec else 0
    return acc, vec


def mcnemar_exact(a, b):
    """Exact binomial McNemar on two 0/1 lists."""
    from scipy import stats
    n01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    if n01 + n10 == 0:
        return 1.0, n01, n10
    p = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue
    return round(float(p), 6), n01, n10


def main():
    base = Path(__file__).resolve().parent.parent.parent
    if not (base / 'POS-Aware-Stemmer').exists() and (base.parent / 'POS-Aware-Stemmer').exists():
        base = base.parent  # repo/scripts layout is one level deeper
    srlex = str(base / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    expanded = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')

    test = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    n = sum(len(s) for s in test)
    print(f"UD Test split: {len(test)} sentences, {n:,} tokens", flush=True)
    print(f"Using expanded dictionary: {Path(expanded).exists()}\n", flush=True)

    configs = [
        ("Full system (all layers)", dict()),
        ("- diacritics restoration", dict(use_diacritics=False)),
        ("- POS information", dict(use_pos=False)),
        ("- lemma corrections", dict(use_corrections=False)),
        ("- participle rule", dict(use_participle_rule=False)),
        ("- hteti rule", dict(use_hteti_rule=False)),
        ("- morph guesser", dict(use_morph_guesser=False)),
        ("- expanded dictionary", dict(use_expanded=False)),
        ("Bare dictionary (no POS, no rules)", dict(use_pos=False, use_corrections=False,
            use_participle_rule=False, use_hteti_rule=False, use_morph_guesser=False,
            use_diacritics=False)),
    ]

    results = []
    full_acc = None
    full_vec = None

    print(f"{'Configuration':<40} {'Accuracy':>8} {'Δ':>8} {'p':>10}", flush=True)
    print(f"{'-'*70}", flush=True)

    for name, kwargs in configs:
        lem = AblationLemmatizer(srlex, expanded_dict_path=expanded, **kwargs)
        acc, vec = evaluate(lem, test)

        if full_acc is None:
            full_acc, full_vec = acc, vec
            delta, p_str, entry = "---", "---", {'config': name, 'accuracy': acc, 'delta': 0}
        else:
            d = acc - full_acc
            delta = f"{d:+.2f} pp"
            p, n01, n10 = mcnemar_exact(vec, full_vec)
            p_str = f"{p:.4g}"
            entry = {'config': name, 'accuracy': acc, 'delta': round(d, 2),
                     'p_exact': p, 'only_this': n01, 'only_full': n10}

        print(f"  {name:<38} {acc:>7.2f}% {delta:>8} {p_str:>10}", flush=True)
        results.append(entry)

        del lem

    # Save
    output = Path(__file__).parent.parent / 'results' / 'ablation_study.json'
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output}", flush=True)


if __name__ == '__main__':
    main()
