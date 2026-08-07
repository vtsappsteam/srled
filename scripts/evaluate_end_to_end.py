#!/usr/bin/env python3
"""
End-to-end evaluation: lemmatization with PREDICTED MSD tags.

Compares:
  1. Gold MSD tags (upper bound - standard evaluation)
  2. Predicted MSD tags from v5 clean model (realistic scenario)

This addresses the reviewer concern about data leakage by using a POS tagger
fine-tuned exclusively on the UD train split.

"""

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path
from collections import defaultdict, Counter


# ============================================================
# Constants (same as pos_lemmatizer_v2.py)
# ============================================================
EKAVIZATION = {
    'također': 'takođe', 'usprkos': 'uprkos', 'unatoč': 'uprkos',
    'željeti': 'želeti', 'vidjeti': 'videti', 'htjeti': 'hteti',
    'razumjeti': 'razumeti', 'smjeti': 'smeti', 'voljeti': 'voleti',
    'živjeti': 'živeti', 'trpjeti': 'trpeti', 'letjeti': 'leteti',
    'gorjeti': 'goreti', 'sjesti': 'sesti',
    'doživiti': 'doživeti', 'voliti': 'voleti', 'resiti': 'rešiti',
}
LEMMA_CORRECTIONS = {
    'premer': 'premijer', 'mišlenje': 'mišljenje', 'skopje': 'skoplje',
    'k': 'ka', 'tko': 'ko', 'netko': 'neko', 'nitko': 'niko',
    'cel': 'ceo',
    'efekt': 'efekat', 'projekt': 'projekat', 'objekt': 'objekat',
    'ambient': 'ambijent',
    'ekonomist': 'ekonomista', 'terorist': 'terorista',
    'gardist': 'gardista', 'alijas': 'alijansa',
    'nauk': 'nauka',
}
_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


# ============================================================
# Lemmatizer (same logic as v2)
# ============================================================
class Lemmatizer:
    def __init__(self, srlex_path, expanded_dict_path=None, v6_restorer=None):
        # Layer 0 = puni v6 modul (Sekcija 4.3); deljen ili kreiran ovde
        if v6_restorer is not None:
            self._v6 = v6_restorer
        else:
            sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
            from v6_restorer import V6Restorer
            self._v6 = V6Restorer(srlex_path, extra_dict_path=expanded_dict_path)
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
                if len(p) < 7:
                    continue
                w, l, m = p[0].lower(), p[1].lower(), p[2]
                l = EKAVIZATION.get(l, l)
                try:
                    fr = int(p[6])
                except:
                    fr = 0
                pos = m[0] if m else '?'
                ex = msd_d[w][m]
                if fr > ex[1]:
                    msd_d[w][m] = (l, fr)
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
                    l = EKAVIZATION.get(l, l)
                    pos = m[0] if m else '?'
                    ex = msd_d[wl][m]
                    if fr > ex[1]: msd_d[wl][m] = (l, fr)
                    pos_d[wl][pos][l] += fr
                    word_d[wl][l] += fr

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
        if not lemma:
            return lemma
        lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        w = word.lower()
        pos = msd[0] if msd else ""

        if w in _HTETI_FORMS and pos == 'V':
            return 'hteti'

        # Layer 0: puni modul za restauraciju dijakritika (Sekcija 4.3)
        if w not in self._word_index and not (_DIAC_CHARS & set(word)):
            restored = self._v6.restore(word, msd)
            if restored != word:
                w = restored.lower()

        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma:
                return self._correct(lemma)

        if pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            if msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')):
                        return self._correct(l)
            for l, f in cands:
                if l == w:
                    return w
            return self._correct(cands[0][0])

        if msd and w in self._msd_index:
            for plen in range(len(msd) - 1, 0, -1):
                for fm, l in self._msd_index[w].items():
                    if fm.startswith(msd[:plen]):
                        return self._correct(l)

        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w:
                    return w
            return self._correct(self._word_index[w][0][0])

        if pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guess = w[:-1]
                if guess in self._word_index:
                    return guess
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guess = w[:-1] + 'a'
                if guess in self._word_index:
                    return guess

        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word
        return w


def load_conllu(path):
    sentences = []
    current = []
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
            current.append((p[1], p[2], p[4]))  # word, lemma, xpos
    if current:
        sentences.append(current)
    return sentences


def main():
    base = Path(__file__).parent.parent.parent
    srlex = str(base / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    v5_model = base / 'NLTK Treniranje' / 'models_v5' / 'perceptron-tagger-ud-clean.pickle'
    test_path = '/tmp/UD_Serbian-SET/sr_set-ud-test.conllu'

    print("=" * 60)
    print("END-TO-END EVALUATION: GOLD vs PREDICTED MSD TAGS")
    print("=" * 60)

    # Load lemmatizer
    expanded = str(base / 'Diacritics-Restoration' / 'data' / 'expanded_supplement.json')
    print(f"\n1. Loading lemmatizer (srLex v1.3 + expanded dict)...", flush=True)
    lem = Lemmatizer(srlex, expanded_dict_path=expanded)
    print("   Done.", flush=True)

    # Load test data
    print("\n2. Loading UD test split...", flush=True)
    gold = load_conllu(test_path)
    n = sum(len(s) for s in gold)
    print(f"   {len(gold)} sentences, {n:,} tokens", flush=True)

    # === Evaluation 1: Gold MSD tags ===
    print("\n3. Evaluation with GOLD MSD tags:", flush=True)
    correct_gold = 0
    total = 0
    per_pos_gold = defaultdict(lambda: [0, 0])  # [correct, total]

    for sent in gold:
        for w, gl, msd in sent:
            total += 1
            pred = lem.lemmatize(w, msd)
            pos = msd[0] if msd else '?'
            per_pos_gold[pos][1] += 1
            if pred and gl and pred.lower() == gl.lower():
                correct_gold += 1
                per_pos_gold[pos][0] += 1

    acc_gold = round(100 * correct_gold / total, 2)
    print(f"   Accuracy: {acc_gold}% ({correct_gold}/{total})", flush=True)

    # === Load POS tagger v5 ===
    print("\n4. Loading POS tagger v5 (clean, no leakage)...", flush=True)
    with open(v5_model, 'rb') as f:
        tagger = pickle.load(f)
    print("   Done.", flush=True)

    # Predict MSD tags
    print("\n5. Predicting MSD tags with v5 tagger...", flush=True)
    words_by_sent = [[w for w, l, m in sent] for sent in gold]
    pred_tags = tagger.tag_sents(words_by_sent)

    # === Evaluation 2: Predicted MSD tags ===
    print("\n6. Evaluation with PREDICTED MSD tags:", flush=True)
    correct_pred = 0
    total2 = 0
    per_pos_pred = defaultdict(lambda: [0, 0])

    for sent_gold, sent_tagged in zip(gold, pred_tags):
        for (w, gl, gold_msd), (_, pred_msd) in zip(sent_gold, sent_tagged):
            total2 += 1
            pred_lemma = lem.lemmatize(w, pred_msd)
            gold_pos = gold_msd[0] if gold_msd else '?'
            per_pos_pred[gold_pos][1] += 1
            if pred_lemma and gl and pred_lemma.lower() == gl.lower():
                correct_pred += 1
                per_pos_pred[gold_pos][0] += 1

    acc_pred = round(100 * correct_pred / total2, 2)
    print(f"   Accuracy: {acc_pred}% ({correct_pred}/{total2})", flush=True)

    # === Summary ===
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Gold MSD tags:      {acc_gold}%")
    print(f"  Predicted MSD tags: {acc_pred}%")
    print(f"  Difference:         {acc_pred - acc_gold:+.2f} pp")
    print(f"  POS tagger MSD acc: 90.89%")
    print()

    # Per-POS comparison
    print(f"  {'POS':<6} {'N':>6} {'Gold':>8} {'Pred':>8} {'Δ':>8}")
    print(f"  {'-'*38}")
    pos_order = ['N', 'V', 'A', 'P', 'R', 'S', 'C', 'M', 'Q']
    for pos in pos_order:
        if pos in per_pos_gold:
            g_c, g_t = per_pos_gold[pos]
            p_c, p_t = per_pos_pred[pos]
            g_acc = round(100 * g_c / g_t, 2) if g_t else 0
            p_acc = round(100 * p_c / p_t, 2) if p_t else 0
            delta = p_acc - g_acc
            print(f"  {pos:<6} {g_t:>6} {g_acc:>7.2f}% {p_acc:>7.2f}% {delta:>+7.2f}")

    print(f"\n{'='*60}")


if __name__ == '__main__':
    main()
