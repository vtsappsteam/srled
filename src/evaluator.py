"""
Evaluation framework for diacritics restoration.

Automatic evaluation: take clean text, strip diacritics,
restore, and compare against the original.

Metrics:
    - Word-level accuracy (entire word correct or not)
    - Character-level accuracy (per-character)
    - Accuracy on words that HAVE diacritics only
    - Accuracy on AMBIGUOUS words only (hardest cases)
    - Per-character precision/recall/F1 for each diacritic pair (c->c-caron/c-acute, s->s-caron, z->z-caron)
    - Confusion matrix for ambiguous words

"""

import os
import re
import json
import time
from collections import defaultdict, Counter

from diacritics import strip_diacritics, has_diacritics, DIACRITIC_CHARS


class DiacriticsEvaluator:
    """Evaluation of diacritics restoration systems."""

    def __init__(self):
        self.results = {}

    def evaluate(self, gold_texts, system_func, method_name='system',
                 domain='unknown'):
        """
        Evaluates a system on a list of texts.

        Args:
            gold_texts: list of sentences WITH diacritics (gold standard)
            system_func: function(text_without_diacritics) -> text_with_diacritics
            method_name: method name (for reporting)
            domain: text domain (news, twitter, web)

        Returns:
            Dict with all metrics
        """
        print(f"\nEvaluation: {method_name} on {domain}")
        print(f"  Sentences: {len(gold_texts):,}")

        start = time.time()

        # Metrics
        total_words = 0
        correct_words = 0
        total_diac_words = 0       # words that HAVE diacritics in gold
        correct_diac_words = 0
        total_ambig_words = 0      # ambiguous words
        correct_ambig_words = 0
        total_chars = 0
        correct_chars = 0
        total_diac_chars = 0       # characters that are diacritic in gold
        correct_diac_chars = 0

        # Per-character confusion: (gold_char, pred_char) -> count
        char_confusion = Counter()

        # Per-word error examples
        errors = []

        # Ambiguous words (for detailed analysis)
        ambiguous_results = defaultdict(lambda: {'correct': 0, 'wrong': 0, 'examples': []})

        for sent_idx, gold_text in enumerate(gold_texts):
            # Strip diacritics
            stripped = strip_diacritics(gold_text)

            # Restore
            restored = system_func(stripped)

            # Tokenize for comparison
            gold_words = self._tokenize_words(gold_text)
            restored_words = self._tokenize_words(restored)

            # Align lengths (may differ due to special character handling)
            min_len = min(len(gold_words), len(restored_words))

            for i in range(min_len):
                gw = gold_words[i]
                rw = restored_words[i]

                if not re.match(r'\w+$', gw, re.UNICODE):
                    continue  # skip punctuation

                total_words += 1

                # Word-level
                if gw == rw:
                    correct_words += 1
                elif len(errors) < 200:
                    errors.append({
                        'gold': gw,
                        'predicted': rw,
                        'stripped': strip_diacritics(gw),
                        'sentence_idx': sent_idx,
                    })

                # Words with diacritics
                if has_diacritics(gw):
                    total_diac_words += 1
                    if gw == rw:
                        correct_diac_words += 1

                # Character-level
                gw_stripped = strip_diacritics(gw)
                for j, (gc, rc) in enumerate(zip(gw, rw)):
                    if j < len(gw) and j < len(rw):
                        total_chars += 1
                        if gc == rc:
                            correct_chars += 1
                        if gc in DIACRITIC_CHARS:
                            total_diac_chars += 1
                            if gc == rc:
                                correct_diac_chars += 1
                            char_confusion[(gc, rc)] += 1

        elapsed = time.time() - start

        # Compute metrics
        result = {
            'method': method_name,
            'domain': domain,
            'num_sentences': len(gold_texts),
            'time_seconds': round(elapsed, 2),

            'word_accuracy': round(100 * correct_words / total_words, 2) if total_words else 0,
            'total_words': total_words,
            'correct_words': correct_words,

            'diac_word_accuracy': round(100 * correct_diac_words / total_diac_words, 2) if total_diac_words else 0,
            'total_diac_words': total_diac_words,
            'correct_diac_words': correct_diac_words,

            'char_accuracy': round(100 * correct_chars / total_chars, 2) if total_chars else 0,
            'total_chars': total_chars,

            'diac_char_accuracy': round(100 * correct_diac_chars / total_diac_chars, 2) if total_diac_chars else 0,
            'total_diac_chars': total_diac_chars,

            'errors_sample': errors[:50],
        }

        # Per-character F1
        char_metrics = self._compute_char_metrics(char_confusion)
        result['per_char_metrics'] = char_metrics

        # Print summary
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Word accuracy (all words):     {result['word_accuracy']:.2f}%")
        print(f"  Word accuracy (diac. words):    {result['diac_word_accuracy']:.2f}%")
        print(f"  Character accuracy:             {result['char_accuracy']:.2f}%")
        print(f"  Diac char accuracy:             {result['diac_char_accuracy']:.2f}%")

        if char_metrics:
            print(f"  Per-character:")
            for char, metrics in sorted(char_metrics.items()):
                print(f"    {char}: P={metrics['precision']:.1f}% "
                      f"R={metrics['recall']:.1f}% F1={metrics['f1']:.1f}%")

        if errors:
            print(f"  Error examples (first 10):")
            for e in errors[:10]:
                print(f"    '{e['stripped']}' -> '{e['predicted']}' "
                      f"(expected: '{e['gold']}')")

        self.results[f"{method_name}_{domain}"] = result
        return result

    def _tokenize_words(self, text):
        """Extracts words from text."""
        return re.findall(r'\S+', text)

    def _compute_char_metrics(self, confusion):
        """
        Computes precision/recall/F1 for each diacritic character.

        For each pair (c-caron/c, c-acute/c, s-caron/s, z-caron/z, d-stroke/dj):
        - Precision: of all predicted c-caron, how many are truly c-caron
        - Recall: of all true c-caron, how many are predicted as c-caron
        """
        diac_chars = ['č', 'ć', 'š', 'ž', 'đ']
        metrics = {}

        for dc in diac_chars:
            tp = confusion.get((dc, dc), 0)  # correctly predicted
            fn = sum(confusion.get((dc, other), 0)
                    for other in set(c for (g, c) in confusion if g == dc)
                    if other != dc)
            fp = sum(confusion.get((other, dc), 0)
                    for other in set(g for (g, c) in confusion if c == dc)
                    if other != dc)

            precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if tp + fn + fp > 0:
                metrics[dc] = {
                    'precision': round(precision, 1),
                    'recall': round(recall, 1),
                    'f1': round(f1, 1),
                    'tp': tp,
                    'fn': fn,
                    'fp': fp,
                }

        return metrics

    def compare_methods(self):
        """Compares all evaluated methods."""
        if not self.results:
            print("No results to compare.")
            return

        print(f"\n{'='*70}")
        print(f"METHOD COMPARISON")
        print(f"{'='*70}")
        print(f"{'Method':<30} {'Domain':<10} {'Word%':>7} {'Diac%':>7} {'Char%':>7} {'Time':>6}")
        print(f"{'-'*70}")

        for key, r in sorted(self.results.items()):
            print(f"{r['method']:<30} {r['domain']:<10} "
                  f"{r['word_accuracy']:>6.2f}% "
                  f"{r['diac_word_accuracy']:>6.2f}% "
                  f"{r['char_accuracy']:>6.2f}% "
                  f"{r['time_seconds']:>5.1f}s")

    def save_results(self, filepath):
        """Saves results to JSON."""
        # Remove error examples for cleaner JSON
        clean = {}
        for key, r in self.results.items():
            clean[key] = {k: v for k, v in r.items() if k != 'errors_sample'}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(clean, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved: {filepath}")


def load_setimes_sentences(filepath):
    """Loads SETimes.SR as a list of sentences (gold standard with diacritics)."""
    sentences = []
    current_words = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append(' '.join(current_words))
                    current_words = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                current_words.append(parts[1])

    if current_words:
        sentences.append(' '.join(current_words))

    return sentences


def load_reldi_sentences(filepath):
    """Loads ReLDI as a list of sentences."""
    sentences = []
    current_words = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_words:
                    sentences.append(' '.join(current_words))
                    current_words = []
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                if '-' in parts[0]:  # skip multiword tokens
                    continue
                current_words.append(parts[1])

    if current_words:
        sentences.append(' '.join(current_words))

    return sentences


if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    from candidate_generator import CandidateGenerator
    from pos_disambiguator import POSDisambiguator

    # Load srLex
    srlex_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'srLex_v1.3.gz'
    )
    cg = CandidateGenerator(srlex_path)
    disamb = POSDisambiguator(cg)

    # Load SETimes
    setimes_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'SETimes.SR', 'set.sr.conll'
    )

    if os.path.exists(setimes_path):
        print("\nLoading SETimes.SR...")
        gold_sents = load_setimes_sentences(setimes_path)
        print(f"  {len(gold_sents)} sentences")

        # Evaluate frequency baseline
        evaluator = DiacriticsEvaluator()

        def frequency_restore(text):
            return disamb.restore_text(text, method='frequency')

        evaluator.evaluate(
            gold_sents[:500],  # first 500 for quick test
            frequency_restore,
            method_name='frequency_baseline',
            domain='news'
        )

        # Save
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        os.makedirs(results_dir, exist_ok=True)
        evaluator.save_results(os.path.join(results_dir, 'eval_quick_test.json'))
    else:
        print(f"SETimes not found: {setimes_path}")
