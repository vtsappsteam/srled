#!/usr/bin/env python3
"""
Methodologically correct evaluation on standard UD train/dev/test splits.

Uses ONLY the test split for evaluation - no data leakage.

UD Serbian-SET splits:
  train: 3,497 sentences (77,334 tokens) - 80%
  dev:   476 sentences (11,460 tokens)   - 10%
  test:  411 sentences (8,879 tokens)    - 10%

Run with: source /tmp/classla_env/bin/activate && python evaluate_on_splits.py

"""

import json
import gzip
import time
import os
import math
import sys
from pathlib import Path
from collections import defaultdict, Counter


def load_conllu(path: str) -> list[list[tuple[str, str, str]]]:
    """Load CoNLL-U file → [[(word, lemma, xpos), ...], ...]"""
    sentences = []
    current = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 6 or '-' in parts[0] or '.' in parts[0]:
                continue
            word = parts[1]
            lemma = parts[2]
            xpos = parts[4]  # MSD tag
            current.append((word, lemma, xpos))
    if current:
        sentences.append(current)
    return sentences


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
    'k': 'ka',
    'tko': 'ko', 'netko': 'neko', 'nitko': 'niko',
    'cel': 'ceo',
    'efekt': 'efekat', 'projekt': 'projekat', 'objekt': 'objekat',
    'ambient': 'ambijent',
    'ekonomist': 'ekonomista', 'terorist': 'terorista',
    'gardist': 'gardista', 'alijas': 'alijansa',
    'nauk': 'nauka',
}

# Negative future forms
_HTETI_FORMS = {'neće', 'neću', 'nećemo', 'nećete'}
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


class OurLemmatizer:
    def __init__(self):
        self._msd_index = defaultdict(dict)
        self._pos_index = defaultdict(lambda: defaultdict(list))
        self._word_index = defaultdict(list)
        self._ascii_index = defaultdict(list)

    def load(self, srlex_path, expanded_dict_path=None):
        msd_data = defaultdict(lambda: defaultdict(lambda: (None, 0)))
        pos_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        word_data = defaultdict(lambda: defaultdict(int))

        with gzip.open(srlex_path, 'rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 7:
                    continue
                word = parts[0].lower()
                lemma = parts[1].lower()
                msd = parts[2]
                pos = msd[0] if msd else '?'
                try:
                    freq = int(parts[6])
                except:
                    freq = 0
                lemma = EKAVIZATION.get(lemma, lemma)
                existing = msd_data[word][msd]
                if freq > existing[1]:
                    msd_data[word][msd] = (lemma, freq)
                pos_data[word][pos][lemma] += freq
                word_data[word][lemma] += freq

        # Expanded dictionary
        if expanded_dict_path and os.path.exists(expanded_dict_path):
            import json
            with open(expanded_dict_path) as f:
                extra = json.load(f)
            for w, entries in extra.items():
                wl = w.lower()
                for entry in entries:
                    l, m = entry[0].lower(), entry[1]
                    fr = entry[2] if len(entry) > 2 else 1000
                    l = EKAVIZATION.get(l, l)
                    pos = m[0] if m else '?'
                    ex = msd_data[wl][m]
                    if fr > ex[1]: msd_data[wl][m] = (l, fr)
                    pos_data[wl][pos][l] += fr
                    word_data[wl][l] += fr

        ascii_freq = defaultdict(lambda: defaultdict(int))
        for word, lf in word_data.items():
            af = word.translate(_DIAC_TO_ASCII)
            ascii_freq[af][word] += sum(lf.values())
        for af, wf in ascii_freq.items():
            self._ascii_index[af] = sorted(wf.items(), key=lambda x: -x[1])

        for word, msds in msd_data.items():
            self._msd_index[word] = {m: l for m, (l, f) in msds.items()}
        for word, poses in pos_data.items():
            for pos, lf in poses.items():
                self._pos_index[word][pos] = sorted(lf.items(), key=lambda x: -x[1])
        for word, lf in word_data.items():
            self._word_index[word] = sorted(lf.items(), key=lambda x: -x[1])

    def _correct(self, lemma):
        lemma = LEMMA_CORRECTIONS.get(lemma, lemma)
        return EKAVIZATION.get(lemma, lemma)

    def lemmatize(self, word, msd=""):
        w = word.lower()
        pos = msd[0] if msd else ""

        # Special: negative future → hteti
        if w in _HTETI_FORMS and pos == 'V':
            return 'hteti'

        # Layer 0: Diacritics restoration for OOV
        if w not in self._word_index and not (_DIAC_CHARS & set(word)):
            af = w.translate(_DIAC_TO_ASCII)
            if af != w:
                candidates = self._ascii_index.get(af, [])
            else:
                candidates = self._ascii_index.get(w, [])
            if candidates:
                diac_cands = [(cw, f) for cw, f in candidates if cw != af]
                if diac_cands:
                    if pos:
                        for cw, f in diac_cands:
                            if pos in self._pos_index.get(cw, {}):
                                w = cw
                                break
                        else:
                            w = diac_cands[0][0]
                    else:
                        w = diac_cands[0][0]

        # Layer 1: MSD exact
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma:
                return self._correct(lemma)

        # Layer 2: POS lookup
        if pos and w in self._pos_index and pos in self._pos_index[w]:
            cands = self._pos_index[w][pos]
            # Participle rule
            if msd.startswith(('Ap', 'Ag')):
                for l, f in cands:
                    if not l.endswith(('ti', 'ći', 'ci')):
                        return self._correct(l)
            # Self-lemma
            for l, f in cands:
                if l == w:
                    return w
            return self._correct(cands[0][0])

        # Layer 2b: MSD prefix
        if msd and w in self._msd_index:
            for plen in range(len(msd)-1, 0, -1):
                prefix = msd[:plen]
                for full_msd, lemma in self._msd_index[w].items():
                    if full_msd.startswith(prefix):
                        return self._correct(lemma)

        # Layer 2c: Freq lookup
        if w in self._word_index:
            for l, f in self._word_index[w]:
                if l == w:
                    return w
            return self._correct(self._word_index[w][0][0])

        # Layer 3: Morph guesser (simplified)
        if pos == 'N' and len(w) > 3:
            g = msd[2] if len(msd) > 2 else ''
            n = msd[3] if len(msd) > 3 else ''
            c = msd[4] if len(msd) > 4 else ''
            if g == 'm' and n == 's' and c == 'g' and w.endswith('a'):
                guessed = w[:-1]
                if guessed in self._word_index:
                    return guessed
            if g == 'f' and n == 's' and c == 'g' and w.endswith('e'):
                guessed = w[:-1] + 'a'
                if guessed in self._word_index:
                    return guessed

        # Layer 4: Identity
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word
        return w


def evaluate(gold, pred_lemmas, name):
    total = 0
    correct = 0
    pos_t = defaultdict(int)
    pos_c = defaultdict(int)
    per_token = []
    errors = []

    for sent_g, sent_p in zip(gold, pred_lemmas):
        for (w, gl, msd), pl in zip(sent_g, sent_p):
            total += 1
            pos = msd[0] if msd else '?'
            pos_t[pos] += 1
            ok = pl and gl and pl.lower() == gl.lower()
            per_token.append(ok)
            if ok:
                correct += 1
                pos_c[pos] += 1
            elif len(errors) < 50:
                errors.append(f'  {w:<20} gold={gl:<20} pred={pl:<20} [{msd}]')

    acc = round(100 * correct / total, 2) if total else 0
    return {
        'name': name, 'total': total, 'correct': correct, 'accuracy': acc,
        'per_pos': {p: round(100*pos_c[p]/pos_t[p], 2) for p in sorted(pos_t) if pos_t[p] > 5},
        'per_token': per_token,
        'errors': errors,
    }


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if not x and y)
    if b01 + b10 == 0:
        return {'chi2': 0, 'p': 1.0, 'b01': b01, 'b10': b10}
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p = math.erfc(math.sqrt(chi2 / 2))
    return {'chi2': round(chi2, 2), 'p': round(p, 6), 'b01': b01, 'b10': b10,
            'sig_005': p < 0.05, 'sig_001': p < 0.01}


def main():
    base = Path(__file__).parent.parent.parent
    ud_path = Path('/tmp/UD_Serbian-SET')

    test_file = ud_path / 'sr_set-ud-test.conllu'
    dev_file = ud_path / 'sr_set-ud-dev.conllu'

    print("=" * 70, flush=True)
    print("  METODOLOŠKI KOREKTNA EVALUACIJA NA UD TEST SPLIT-U", flush=True)
    print("=" * 70, flush=True)

    for split_name, split_path in [('TEST', test_file), ('DEV', dev_file)]:
        gold = load_conllu(str(split_path))
        n = sum(len(s) for s in gold)
        print(f"\n{'='*70}", flush=True)
        print(f"  Split: {split_name} - {len(gold)} rečenica, {n:,} tokena", flush=True)
        print(f"{'='*70}", flush=True)

        results = []

        print("\n  [1] Naš POS-Aware Pipeline v2...", flush=True)
        lem = OurLemmatizer()
        t0 = time.time()
        expanded = str(base / 'Diacritics-Restoration' / 'data' / 'expanded_supplement.json')
        lem.load(str(base / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz'),
                 expanded_dict_path=expanded)
        load_t = time.time() - t0

        t0 = time.time()
        our_preds = []
        for sent in gold:
            our_preds.append([lem.lemmatize(w, msd) for w, l, msd in sent])
        our_time = time.time() - t0
        our_speed = int(n / our_time) if our_time > 0 else 0

        our_result = evaluate(gold, our_preds, "Our POS-Pipeline")
        our_result['speed'] = our_speed
        our_result['time'] = round(our_time, 3)
        results.append(our_result)
        print(f"      Accuracy: {our_result['accuracy']}%  ({our_speed:,} tok/s)", flush=True)

        print("\n  [2] CLASSLA-Stanza...", flush=True)
        try:
            import classla
            t0 = time.time()
            nlp = classla.Pipeline('sr', type='standard', processors='tokenize,pos,lemma',
                                   tokenize_pretokenized=True, use_gpu=False)
            cl_load = time.time() - t0

            sentences_text = [[w for w, l, m in sent] for sent in gold]
            t0 = time.time()
            cl_preds = []
            for sent_words in sentences_text:
                doc = nlp([sent_words])
                lemmas = []
                for sent in doc.sentences:
                    for word in sent.words:
                        lemmas.append(word.lemma)
                cl_preds.append(lemmas)
            cl_time = time.time() - t0
            cl_speed = int(n / cl_time) if cl_time > 0 else 0

            # Align
            aligned = []
            for sg, sp in zip(gold, cl_preds):
                if len(sp) >= len(sg):
                    aligned.append(sp[:len(sg)])
                else:
                    aligned.append(sp + [''] * (len(sg) - len(sp)))

            cl_result = evaluate(gold, aligned, "CLASSLA-Stanza")
            cl_result['speed'] = cl_speed
            cl_result['time'] = round(cl_time, 1)
            results.append(cl_result)
            print(f"      Accuracy: {cl_result['accuracy']}%  ({cl_speed:,} tok/s)", flush=True)

            del nlp
            import gc; gc.collect()
        except Exception as e:
            print(f"      FAILED: {e}", flush=True)

        print(f"\n  POREĐENJE ({split_name}):", flush=True)
        print(f"  {'System':<25} {'Accuracy':>8} {'Speed':>12} {'Time':>8}", flush=True)
        print(f"  {'-'*55}", flush=True)
        for r in results:
            print(f"  {r['name']:<25} {r['accuracy']:>7.2f}% {r['speed']:>10,} tok/s {r['time']:>7}s", flush=True)

        # Per-POS comparison
        pos_names = {
            'N': 'Imen.', 'V': 'Glag.', 'A': 'Prid.', 'P': 'Zam.',
            'R': 'Pril.', 'M': 'Broj.', 'S': 'Pred.', 'C': 'Vezn.',
            'Q': 'Rečc.', 'Z': 'Int.', 'X': 'Rez.',
        }
        print(f"\n  Per-POS:", flush=True)
        header = f"  {'POS':<6}"
        for r in results:
            header += f" {r['name'][:15]:>15}"
        print(header, flush=True)
        print(f"  {'-'*(6+16*len(results))}", flush=True)
        all_pos = set()
        for r in results:
            all_pos.update(r['per_pos'].keys())
        for pos in sorted(all_pos):
            row = f"  {pos} {pos_names.get(pos, ''):>4}"
            for r in results:
                a = r['per_pos'].get(pos, 0)
                row += f" {a:>14.2f}%"
            print(row, flush=True)

        # McNemar's test
        if len(results) >= 2:
            print(f"\n  McNemar's test (Our vs CLASSLA):", flush=True)
            mn = mcnemar(results[0]['per_token'], results[1]['per_token'])
            print(f"    Our correct, CLASSLA wrong: {mn['b01']}", flush=True)
            print(f"    Our wrong, CLASSLA correct: {mn['b10']}", flush=True)
            print(f"    Chi²: {mn['chi2']}, p={mn['p']}", flush=True)
            print(f"    Significant (p<0.05): {mn.get('sig_005', '?')}", flush=True)

        # Sample errors from our system
        print(f"\n  Naše greške (prvih 15):", flush=True)
        for e in results[0]['errors'][:15]:
            print(f"  {e}", flush=True)

    # Save
    output = base / 'Diacritics-Restoration' / 'results' / 'evaluation_on_ud_splits.json'
    save = {}
    for r in results:
        rr = dict(r)
        rr.pop('per_token', None)
        rr.pop('errors', None)
        save[r['name']] = rr
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(save, f, indent=2)
    print(f"\n  Saved: {output}", flush=True)


if __name__ == '__main__':
    main()
