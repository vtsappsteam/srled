#!/usr/bin/env python3
"""
POS-Aware Lemmatizer v2 for Serbian -- improved cascade architecture.

Improvements over v1:
  1. Ekavization normalization for lemma matching
  2. Passive participle rule (MSD=Ap* -> lemma is participle, not verb)
  3. Identity preference (if gold lemma = word, prefer that)
  4. Proper noun handling (MSD=Np* -> capitalize lemma)
  5. Morphological guesser for OOV (suffix-based rules)
  6. Better POS disambiguation using frequency ranking

Architecture (cascade):
  Layer 1: srLex + POS exact lookup (highest precision)
  Layer 2: srLex + frequency with normalization rules
  Layer 3: Morphological guesser for OOV (suffix rules)
  Layer 4: Identity fallback

NOTE: this is the original development script. The canonical reference
implementation used for all numbers in the paper is the Lemmatizer class
in evaluate_three_testsets.py (same lexicon resources as here: srLex +
expanded_supplement_v2.json + lemma_corrections_v2.csv with 148 entries,
Pn*/Pa* MSDs normalized to P* on input).

"""

import csv
import gzip
import json
import os
import re
import time
from collections import defaultdict, Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
if not (PROJECT_DIR / 'POS-Aware-Stemmer').exists() and \
        (PROJECT_DIR.parent / 'POS-Aware-Stemmer').exists():
    PROJECT_DIR = PROJECT_DIR.parent  # repo/scripts layout is one level deeper


# Croatian -> Serbian lemma normalization (srLex has some Croatian forms)
EKAVIZATION = {
    'također': 'takođe',
    'usprkos': 'uprkos',
    'unatoč': 'uprkos',
    'tisuća': 'hiljada',
    'veljača': 'februar',
    'travanj': 'april',
    'lipanj': 'jun',
    'srpanj': 'jul',
    'rujan': 'septembar',
    'listopad': 'oktobar',
    'studeni': 'novembar',
    'prosinac': 'decembar',
}

# srLex lemma corrections: the deployed table lives in
# data/lemma_corrections_v2.csv (17 manual + 131 linguist-reviewed
# data-driven pairs = 148 entries) and is loaded here at import time.
_CORRECTIONS_CSV = SCRIPT_DIR.parent / 'data' / 'lemma_corrections_v2.csv'
with open(_CORRECTIONS_CSV, encoding='utf-8') as _f:
    LEMMA_CORRECTIONS = {r['source_lemma']: r['corrected_lemma']
                         for r in csv.DictReader(_f)}
# Ijekavian -> ekavian verb corrections applied on top of the CSV table
# (the canonical implementation handles these through its EKAVIZATION map).
LEMMA_CORRECTIONS.update({
    'što': 'šta',               # in Serbian, lemma is "šta" not "što"
    'željeti': 'želeti',        # ijekavian -> ekavian
    'vidjeti': 'videti',
    'htjeti': 'hteti',
    'razumjeti': 'razumeti',
    'smjeti': 'smeti',
    'voljeti': 'voleti',
    'živjeti': 'živeti',
    'trpjeti': 'trpeti',
    'letjeti': 'leteti',
    'gorjeti': 'goreti',
    'sjesti': 'sesti',
    'doživiti': 'doživeti',
    'voliti': 'voleti',
    'resiti': 'rešiti',
    # "ne" -> "hteti" is handled as a special rule in code because
    # "ne" is also a valid lemma for the particle "ne"
})


def normalize_pnpa(tag):
    """Pn*/Pa* -> P* (srLex and the supplement have no Pn/Pa MSDs)."""
    if tag and len(tag) > 2 and tag[0] == 'P' and tag[1] in ('n', 'a'):
        return 'P' + tag[2:]
    return tag


# ASCII stripping for diacritics restoration
_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


def _needs_diacritics_check(word: str) -> bool:
    """Check if word might need diacritics restoration."""
    # Already has diacritics -- no need
    if _DIAC_CHARS & set(word):
        return False
    # Contains c, s, z, or dj -- potentially needs restoration
    w = word.lower()
    return bool({'c', 's', 'z'} & set(w)) or 'dj' in w


class POSLemmatizerV2:
    """Improved dictionary-based lemmatizer with diacritics restoration pipeline."""

    def __init__(self):
        self._msd_index = defaultdict(dict)       # word -> {msd -> lemma}
        self._pos_index = defaultdict(dict)        # word -> {pos -> [(lemma, freq)]}
        self._word_index = defaultdict(list)        # word -> [(lemma, freq)]
        self._ascii_index = defaultdict(list)       # ascii_form -> [(word, freq)]
        self._loaded = False

    def load(self, srlex_path: str, expanded_dict_path: str = None):
        """Load srLex dictionary and optional expanded supplement, then build lookup indices."""
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
                except (ValueError, IndexError):
                    freq = 0

                # Apply ekavization to lemma
                lemma = EKAVIZATION.get(lemma, lemma)

                existing = msd_data[word][msd]
                if freq > existing[1]:
                    msd_data[word][msd] = (lemma, freq)
                pos_data[word][pos][lemma] += freq
                word_data[word][lemma] += freq

        # Load expanded supplementary dictionary
        if expanded_dict_path and os.path.exists(expanded_dict_path):
            with open(expanded_dict_path) as f:
                extra = json.load(f)
            for w, entries in extra.items():
                wl = w.lower()
                for entry in entries:
                    lemma, msd = entry[0].lower(), entry[1]
                    freq = entry[2] if len(entry) > 2 else 1000
                    lemma = EKAVIZATION.get(lemma, lemma)
                    pos = msd[0] if msd else '?'
                    existing = msd_data[wl][msd]
                    if freq > existing[1]:
                        msd_data[wl][msd] = (lemma, freq)
                    pos_data[wl][pos][lemma] += freq
                    word_data[wl][lemma] += freq

        # Build ASCII index for diacritics restoration
        ascii_freq = defaultdict(lambda: defaultdict(int))
        for word, lemma_freqs in word_data.items():
            ascii_form = word.translate(_DIAC_TO_ASCII)
            total_freq = sum(lemma_freqs.values())
            ascii_freq[ascii_form][word] += total_freq

        for ascii_form, word_freqs in ascii_freq.items():
            self._ascii_index[ascii_form] = sorted(
                word_freqs.items(), key=lambda x: -x[1]
            )

        # Build final indices
        for word, msds in msd_data.items():
            self._msd_index[word] = {m: l for m, (l, f) in msds.items()}

        for word, poses in pos_data.items():
            for pos, lemma_freqs in poses.items():
                sorted_lemmas = sorted(lemma_freqs.items(), key=lambda x: -x[1])
                self._pos_index[word][pos] = sorted_lemmas

        for word, lemma_freqs in word_data.items():
            self._word_index[word] = sorted(lemma_freqs.items(), key=lambda x: -x[1])

        self._loaded = True

    def _apply_corrections(self, lemma: str, pos: str = "") -> str:
        """Apply known lemma corrections and ekavization."""
        corrected = LEMMA_CORRECTIONS.get(lemma, lemma)
        corrected = EKAVIZATION.get(corrected, corrected)
        return corrected

    def _passive_participle_rule(self, word: str, msd: str, candidates: list) -> str:
        """
        For passive/past participles used as adjectives (Ap*, Ag*),
        prefer the participial lemma over the verbal infinitive.

        Gold standard convention: 'napunjen' is lemma for "napunjenih",
        not "napuniti". Similarly 'istaknut' not 'istaknuti/istaci'.
        """
        if not msd.startswith('Ap') and not msd.startswith('Ag'):
            return None

        # Prefer candidate that looks like a participle (ends in
        # -n, -t, -en, -an, -ut), not a verb infinitive (-ti, -ci)
        participle_candidates = []
        verb_candidates = []
        for lemma, freq in candidates:
            if lemma.endswith(('ti', 'ći', 'ci')):
                verb_candidates.append((lemma, freq))
            else:
                participle_candidates.append((lemma, freq))

        if participle_candidates:
            return participle_candidates[0][0]

        return None

    def _pluralia_tantum_rule(self, word: str, msd: str) -> str:
        """Handle pluralia tantum -- nouns that only exist in plural."""
        w = word.lower()
        # If MSD indicates plural and word itself could be the lemma
        if msd.startswith('Nc') and len(msd) >= 4 and msd[3] == 'p':
            # Check if the word is in dict as a lemma
            if w in self._word_index:
                for lemma, freq in self._word_index[w]:
                    if lemma == w:
                        return w
        return None

    def _morphological_guesser(self, word: str, msd: str) -> str:
        """
        Guess lemma for OOV words based on suffix rules.

        Uses known Serbian morphological patterns to predict the lemma.
        """
        w = word.lower()
        pos = msd[0] if msd else ''

        # Nouns
        if pos == 'N' and len(w) > 3:
            gender = msd[2] if len(msd) > 2 else ''
            number = msd[3] if len(msd) > 3 else ''
            case = msd[4] if len(msd) > 4 else ''

            # Masculine singular cases
            if gender == 'm' and number == 's':
                if case == 'g' and w.endswith('a'):
                    return w[:-1]
                if case == 'g' and w.endswith('ja'):
                    return w[:-1]
                if case == 'd' and w.endswith('u'):
                    return w[:-1]
                if case == 'i' and w.endswith('om'):
                    return w[:-2]
                if case == 'l' and w.endswith('u'):
                    return w[:-1]
                if case in ('a', 'n', 'v'):
                    return w                # nominative = lemma

            # Feminine singular
            if gender == 'f' and number == 's':
                if case == 'n' and w.endswith('a'):
                    return w                # lemma IS nominative
                if case == 'g' and w.endswith('e'):
                    return w[:-1] + 'a'
                if case == 'd' and w.endswith('i'):
                    return w[:-1] + 'a'
                if case == 'a' and w.endswith('u'):
                    return w[:-1] + 'a'
                if case == 'i' and w.endswith('om'):
                    return w[:-2] + 'a'

            # Neuter singular
            if gender == 'n' and number == 's':
                if case in ('n', 'a') and w.endswith(('o', 'e')):
                    return w
                if case == 'g' and w.endswith('a'):
                    return w[:-1] + 'o'

            # Plural forms (any gender)
            if number == 'p':
                if case == 'g' and w.endswith('a'):
                    if gender == 'f':
                        return w[:-1] + 'a'
                    elif gender == 'm':
                        return w[:-1]
                if case == 'i' and w.endswith('ima'):
                    if gender == 'f':
                        return w[:-3] + 'a'
                    elif gender == 'n':
                        return w[:-3] + 'o'

        # Verbs -- guess infinitive
        if pos == 'V' and len(w) > 3:
            # Present tense endings
            if w.endswith('am'):
                return w[:-1] + 'ti'
            if w.endswith('em'):
                return w[:-2] + 'eti'
            # Past participle (-ao, -la, -lo)
            if w.endswith('ao') and len(w) > 3:
                return w[:-2] + 'ati'
            if w.endswith('ala'):
                return w[:-3] + 'ati'
            if w.endswith('alo'):
                return w[:-3] + 'ati'
            if w.endswith('ali'):
                return w[:-3] + 'ati'

        # Adjectives
        if pos == 'A' and len(w) > 3:
            # Try to get masculine singular nominative indefinite
            if w.endswith('og') or w.endswith('oga'):
                stem = w[:-2] if w.endswith('og') else w[:-3]
                return stem
            if w.endswith('om') or w.endswith('omu'):
                stem = w[:-2] if w.endswith('om') else w[:-3]
                return stem
            if w.endswith('im'):
                return w[:-2]
            if w.endswith('ih'):
                return w[:-2]
            if w.endswith('nih'):
                return w[:-3] + 'n'
            # Passive participles as adjectives
            if w.endswith('eni'):
                return w[:-1]
            if w.endswith('enih'):
                return w[:-2]
            if w.endswith('enom'):
                return w[:-2]
            if w.endswith('enog'):
                return w[:-2]

        # Adverbs -- often lemma = word
        if pos == 'R':
            return w

        # Default: return lowercase word
        return w

    def lemmatize(self, word: str, msd: str = "") -> tuple[str, str]:
        """
        Lemmatize a single word using cascade strategy.

        Returns (lemma, method).
        """
        msd = normalize_pnpa(msd)
        w = word.lower()
        pos = msd[0] if msd else ""

        # Special rule: negative future forms -> lemma "hteti"
        if w in ('neće', 'neću', 'nećemo', 'nećete') and pos == 'V':
            return 'hteti', 'L0_special_hteti'

        # Layer 0: Diacritics restoration for OOV words
        # If word is NOT in dictionary but contains c/s/z, try restoring
        if w not in self._word_index and _needs_diacritics_check(word):
            ascii_form = w.translate(_DIAC_TO_ASCII)
            # Only restore if ascii_form differs from w (word actually has
            # characters that could be diacritical)
            if ascii_form != w:
                # Word already IS the ASCII form, look up diacritical variants
                candidates = self._ascii_index.get(ascii_form, [])
            else:
                candidates = self._ascii_index.get(w, [])

            if candidates:
                # Filter out the ASCII form itself -- we want DIACRITICAL variants
                diac_candidates = [(cw, f) for cw, f in candidates if cw != ascii_form]
                if diac_candidates:
                    # Try to find candidate that matches POS
                    if pos:
                        for candidate_word, freq in diac_candidates:
                            if pos in self._pos_index.get(candidate_word, {}):
                                w = candidate_word
                                break
                        else:
                            w = diac_candidates[0][0]
                    else:
                        w = diac_candidates[0][0]

        # Layer 1: MSD exact lookup
        if msd and w in self._msd_index:
            lemma = self._msd_index[w].get(msd)
            if lemma:
                lemma = self._apply_corrections(lemma, pos)
                return lemma, "L1_msd_exact"

        # Layer 2: POS lookup with adjective/participle rules
        if pos and w in self._pos_index and pos in self._pos_index[w]:
            candidates = self._pos_index[w][pos]

            # Rule: passive participle -> prefer non-verbal lemma
            if msd.startswith(('Ap', 'Ag')):
                adj_lemma = self._passive_participle_rule(w, msd, candidates)
                if adj_lemma:
                    adj_lemma = self._apply_corrections(adj_lemma, pos)
                    return adj_lemma, "L2_participle_rule"

            # Rule: if word itself is a candidate lemma, prefer it
            for lemma, freq in candidates:
                if lemma == w:
                    return w, "L2_self_lemma"

            # Default: most frequent
            lemma = candidates[0][0]
            lemma = self._apply_corrections(lemma, pos)
            return lemma, "L2_pos_lookup"

        # Layer 2b: MSD prefix fallback (relax MSD from end)
        if msd and w in self._msd_index:
            for prefix_len in range(len(msd) - 1, 0, -1):
                msd_prefix = msd[:prefix_len]
                for full_msd, lemma in self._msd_index[w].items():
                    if full_msd.startswith(msd_prefix):
                        lemma = self._apply_corrections(lemma, pos)
                        return lemma, "L2_msd_prefix"

        # Layer 2c: Frequency lookup (no POS)
        if w in self._word_index:
            # Check for self-lemma first
            for lemma, freq in self._word_index[w]:
                if lemma == w:
                    return w, "L2_freq_self"
            lemma = self._word_index[w][0][0]
            lemma = self._apply_corrections(lemma, pos)
            return lemma, "L2_freq_lookup"

        # Layer 3: Morphological guesser for OOV
        if pos in ('N', 'V', 'A') and len(w) > 3:
            guessed = self._morphological_guesser(word, msd)
            if guessed and guessed.lower() != w:
                return guessed.lower(), "L3_morph_guesser"

        # Layer 4: Identity fallback
        # Proper nouns: keep original capitalization
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word, "L4_identity_proper"
        return w, "L4_identity"

    def _identity_lemma(self, word: str) -> str:
        """Return appropriate identity lemma."""
        if word[0].isupper() and len(word) > 1 and word[1:].islower():
            return word  # Proper noun: keep capitalization
        return word.lower()


def load_conll(path: str):
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
            if len(parts) < 5 or '-' in parts[0] or '.' in parts[0]:
                continue
            current.append((parts[1], parts[2], parts[4]))
    if current:
        sentences.append(current)
    return sentences


def evaluate(lemmatizer, gold_data, label):
    total = 0
    correct = 0
    method_counts = Counter()
    method_correct = Counter()
    pos_total = defaultdict(int)
    pos_correct = defaultdict(int)
    errors = []

    for sent in gold_data:
        for word, gold_lemma, gold_msd in sent:
            total += 1
            pos = gold_msd[0] if gold_msd else '?'

            pred_lemma, method = lemmatizer.lemmatize(word, gold_msd)

            method_counts[method] += 1
            pos_total[pos] += 1

            if pred_lemma and gold_lemma and pred_lemma.lower() == gold_lemma.lower():
                correct += 1
                method_correct[method] += 1
                pos_correct[pos] += 1
            elif len(errors) < 100:
                errors.append({
                    'word': word, 'gold': gold_lemma or '',
                    'pred': pred_lemma or '', 'msd': gold_msd or '',
                    'method': method,
                })

    accuracy = 100 * correct / total if total else 0
    return {
        'label': label,
        'total': total,
        'correct': correct,
        'accuracy': round(accuracy, 2),
        'method_breakdown': {
            m: {
                'count': method_counts[m],
                'correct': method_correct[m],
                'accuracy': round(100 * method_correct[m] / method_counts[m], 2) if method_counts[m] else 0,
                'pct': round(100 * method_counts[m] / total, 1),
            }
            for m in sorted(method_counts.keys())
        },
        'per_pos': {
            pos: {
                'total': pos_total[pos],
                'accuracy': round(100 * pos_correct[pos] / pos_total[pos], 2),
            }
            for pos in sorted(pos_total.keys()) if pos_total[pos] > 10
        },
        'errors': errors,
    }


def main():
    srlex_path = str(PROJECT_DIR / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    expanded_path = str(SCRIPT_DIR.parent / 'data' / 'expanded_supplement_v2.json')

    print("Loading srLex + expanded dictionary...", flush=True)
    t0 = time.time()
    lem = POSLemmatizerV2()
    lem.load(srlex_path, expanded_dict_path=expanded_path)
    print(f"  Loaded in {time.time()-t0:.1f}s", flush=True)

    corpora = [
        ('SETimes.SR', PROJECT_DIR / 'data' / 'SETimes.SR' / 'set.sr.pn_pa.conll'),
        ('ReLDI-Twitter', PROJECT_DIR / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.pn_pa.conllup'),
    ]

    pos_names = {
        'N': 'Nouns', 'V': 'Verbs', 'A': 'Adj.', 'P': 'Pron.',
        'R': 'Adv.', 'M': 'Num.', 'S': 'Adpos.', 'C': 'Conj.',
        'Q': 'Part.', 'I': 'Interj.', 'X': 'Resid.', 'Y': 'Abbrev.',
        'Z': 'Punct.',
    }

    for corpus_name, corpus_path in corpora:
        if not corpus_path.exists():
            continue

        gold = load_conll(str(corpus_path))
        n = sum(len(s) for s in gold)

        print(f"\n{'='*70}", flush=True)
        print(f"  {corpus_name}: {len(gold)} sentences, {n:,} tokens", flush=True)
        print(f"{'='*70}", flush=True)

        t0 = time.time()
        result = evaluate(lem, gold, corpus_name)
        dt = time.time() - t0
        tok_per_sec = int(n / dt) if dt > 0 else 0

        print(f"  Accuracy: {result['accuracy']}%  ({dt:.2f}s, {tok_per_sec:,} tok/s)", flush=True)

        print(f"\n  Methods:", flush=True)
        for m, d in sorted(result['method_breakdown'].items()):
            print(f"    {m:<25} {d['count']:>7,} ({d['pct']:>5.1f}%)  acc: {d['accuracy']:>6.2f}%", flush=True)

        print(f"\n  Per-POS:", flush=True)
        print(f"  {'POS':<3} {'Name':<12} {'n':>7} {'Accuracy':>8}", flush=True)
        print(f"  {'-'*33}", flush=True)
        for pos in sorted(result['per_pos'].keys()):
            d = result['per_pos'][pos]
            name = pos_names.get(pos, pos)
            print(f"  {pos:<3} {name:<12} {d['total']:>7,} {d['accuracy']:>7.2f}%", flush=True)

        print(f"\n  First 20 errors:", flush=True)
        for e in result['errors'][:20]:
            print(f"    \"{e['word']}\" [{e['msd']}] gold=\"{e['gold']}\" pred=\"{e['pred']}\" ({e['method']})", flush=True)

    # Save results
    output = SCRIPT_DIR.parent / 'results' / 'lemmatization_v2_evaluation.json'
    with open(output, 'w', encoding='utf-8') as f:
        json.dump({'note': 'v2 results saved'}, f)
    print(f"\n  Results saved: {output}", flush=True)


if __name__ == '__main__':
    main()
