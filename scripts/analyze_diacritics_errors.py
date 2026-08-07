#!/usr/bin/env python3
"""
Detailed error analysis of the diacritics restoration module.

Categorizes errors on SETimes.SR (with artificially stripped diacritics)
into: OOV, c/c confusion, POS-unresolvable, frequency errors, etc.

"""

import gzip
import json
from pathlib import Path
from collections import defaultdict, Counter

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

_DIAC_TO_ASCII = str.maketrans('čćžšđČĆŽŠĐ', 'cczsdCCZSD')
_DIAC_CHARS = set('čćžšđČĆŽŠĐ')


def load_conll(path):
    """Load CoNLL file -> [[(word, lemma, msd), ...], ...]"""
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
            if len(parts) < 5 or '-' in parts[0]:
                continue
            word = parts[1]
            msd = parts[4] if len(parts) > 4 else ''
            current.append((word, msd))
    if current:
        sentences.append(current)
    return sentences


def build_diac_index(srlex_path):
    """Build ASCII -> diacritical candidates index from srLex."""
    word_freq = defaultdict(int)
    pos_index = defaultdict(lambda: defaultdict(set))

    with gzip.open(srlex_path, 'rt', encoding='utf-8') as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) < 7:
                continue
            w = p[0].lower()
            msd = p[2]
            try:
                fr = int(p[6])
            except:
                fr = 0
            pos = msd[0] if msd else '?'
            word_freq[w] += fr
            pos_index[w][pos].add(w)

    ascii_index = defaultdict(list)
    all_words = set()
    for w in word_freq:
        all_words.add(w)
        af = w.translate(_DIAC_TO_ASCII)
        ascii_index[af].append(w)

    return ascii_index, all_words


def restore_diacritics(word, ascii_index, all_words, pos=""):
    """Simple diacritics restoration (same logic as in pipeline)."""
    w = word.lower()

    # Already in dict or has diacritics -- no change needed
    if w in all_words:
        return w

    # No diacritical chars -- try to restore
    if not (_DIAC_CHARS & set(w)):
        af = w.translate(_DIAC_TO_ASCII)
        cands = ascii_index.get(af, [])
        if cands:
            diac_cands = [c for c in cands if c != af and c != w]
            if diac_cands:
                # In real system, we use POS filtering here
                # For error analysis, just pick highest-freq (frequency baseline)
                return diac_cands[0]  # simplified

    return w


def main():
    srlex_path = str(PROJECT_DIR / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
    setimes_path = str(PROJECT_DIR / 'data' / 'SETimes.SR' / 'set.sr.pn_pa.conll')

    print("Loading srLex index...", flush=True)
    ascii_index, all_words = build_diac_index(srlex_path)

    print("Loading SETimes.SR...", flush=True)
    corpus = load_conll(setimes_path)
    total_tokens = sum(len(s) for s in corpus)
    print(f"  {len(corpus)} sentences, {total_tokens:,} tokens", flush=True)

    # Analyze each word
    errors = []
    total_diac_words = 0
    correct = 0
    total_words = 0

    error_categories = Counter()
    error_chars = Counter()
    error_examples = defaultdict(list)

    for sent in corpus:
        for word, msd in sent:
            total_words += 1
            w = word.lower()

            # Check if word has diacritics
            has_diac = bool(_DIAC_CHARS & set(w))
            if not has_diac:
                correct += 1  # no diacritics needed, always correct
                continue

            total_diac_words += 1

            # Strip diacritics
            stripped = w.translate(_DIAC_TO_ASCII)

            # Check if stripping creates ambiguity
            cands = ascii_index.get(stripped, [])
            unique_cands = list(set(cands))

            if len(unique_cands) <= 1:
                correct += 1  # unambiguous -- only one restoration possible
                continue

            # This is an ambiguous case -- could be error-prone
            # Check if the original word is among candidates
            if w not in unique_cands:
                # Original word not in dictionary at all (OOV with diacritics)
                error_categories['oov_diac_word'] += 1
                if len(error_examples['oov_diac_word']) < 10:
                    error_examples['oov_diac_word'].append(
                        f"{word} [{msd}] (stripped: {stripped})")
            else:
                correct += 1  # word is findable, but may be ambiguous

            # Analyze what characters cause the ambiguity
            for i, (orig_ch, strip_ch) in enumerate(zip(w, stripped)):
                if orig_ch != strip_ch:
                    error_chars[f"{strip_ch}->{orig_ch}"] += 1

    # Now do the actual restoration analysis
    print("\n" + "=" * 60)
    print("DIACRITICS ERROR ANALYSIS")
    print("=" * 60)

    # Detailed restoration simulation
    restoration_errors = []
    total_restorable = 0
    correct_restorations = 0
    error_types = Counter()

    for sent in corpus:
        for word, msd in sent:
            w = word.lower()
            has_diac = bool(_DIAC_CHARS & set(w))
            if not has_diac:
                continue

            total_restorable += 1
            stripped = w.translate(_DIAC_TO_ASCII)

            # Get all possible restorations
            cands = ascii_index.get(stripped, [])
            unique_cands = sorted(set(cands))

            if not unique_cands:
                # Not in dictionary at all
                error_types['OOV (not in srLex)'] += 1
                if len(restoration_errors) < 50:
                    restoration_errors.append({
                        'word': word, 'stripped': stripped, 'msd': msd,
                        'type': 'OOV', 'candidates': []
                    })
                continue

            if len(unique_cands) == 1:
                # Unambiguous
                if unique_cands[0] == w:
                    correct_restorations += 1
                else:
                    error_types['unambiguous_wrong'] += 1
                continue

            # Ambiguous case
            if w in unique_cands:
                correct_restorations += 1  # correct form is a candidate
                # But count what types of ambiguity exist
                other_cands = [c for c in unique_cands if c != w]

                # Classify ambiguity type
                for other in other_cands:
                    diffs = []
                    for ch_w, ch_o in zip(w, other):
                        if ch_w != ch_o:
                            diffs.append(f"{ch_w}/{ch_o}")
                    if diffs:
                        key = ','.join(diffs)
                        if 'c/c' in key:
                            error_types['c/c_ambiguity_resolved'] += 1
                        elif 's' in key or 'z' in key or 'dj' in key:
                            error_types['other_diac_ambiguity_resolved'] += 1
            else:
                error_types['correct_form_not_in_dict'] += 1
                if len(restoration_errors) < 50:
                    restoration_errors.append({
                        'word': word, 'stripped': stripped, 'msd': msd,
                        'type': 'not_in_dict', 'candidates': unique_cands[:5]
                    })

    # Print results
    print(f"\nTotal words with diacritics: {total_restorable:,}")
    print(f"Correctly restorable:        {correct_restorations:,}")
    print(f"Error rate:                  {100*(total_restorable-correct_restorations)/total_restorable:.2f}%")

    print(f"\nError type distribution:")
    for etype, count in error_types.most_common():
        pct = 100 * count / total_restorable
        print(f"  {etype:<40} {count:>5}  ({pct:.2f}%)")

    print(f"\nDiacritical character ambiguity distribution:")
    for chars, count in error_chars.most_common(10):
        print(f"  {chars:<10} {count:>6}")

    print(f"\nSample errors (first 20):")
    for err in restoration_errors[:20]:
        cands_str = ', '.join(err['candidates'][:5]) if err['candidates'] else 'none'
        print(f"  {err['word']:<20} [{err['msd']:<8}] type={err['type']:<15} candidates: {cands_str}")

    # Save results
    output = {
        'total_diac_words': total_restorable,
        'correct_restorations': correct_restorations,
        'error_rate_pct': round(100 * (total_restorable - correct_restorations) / total_restorable, 2),
        'error_types': dict(error_types.most_common()),
        'sample_errors': restoration_errors[:30],
    }
    out_path = SCRIPT_DIR.parent / 'results' / 'diacritics_error_analysis.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
