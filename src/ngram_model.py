"""
N-gram contextual model for diacritics disambiguation.

Uses bigram and trigram probabilities from the srWaC corpus
to resolve cases where neither the dictionary nor POS tags suffice.

Example: "nece" -> "nece" or "nece"?
  - Bigram P("nece" | previous_word) >> P("nece" | previous_word)
  - Therefore we pick the correct diacritized form.

Example: "vrste reci" -> "vrste reci" (noun) or "vrste reci" (verb)?
  - Bigram P("reci" | "vrste") >> P("reci" | "vrste")
  - Therefore we pick the noun form.

Trained from the srWaC corpus or any text WITH diacritics.

"""

import os
import csv
import gzip
import time
import math
import json
import pickle
from collections import defaultdict, Counter

from diacritics import strip_diacritics


class NgramModel:
    """
    Bigram language model for contextual diacritics disambiguation.

    Stores P(word | prev_word) probabilities for fast candidate scoring.
    """

    def __init__(self):
        self.bigram_counts = defaultdict(Counter)  # prev -> {word: count}
        self.unigram_counts = Counter()
        self.total_tokens = 0
        self._loaded = False

    def train_from_srwac(self, srwac_path, max_sentences=None):
        """
        Trains the bigram model from an srWaC CSV file.

        Args:
            srwac_path: path to srWaC CSV (Token,Lemma,MSD,Sentence_ID)
            max_sentences: maximum number of sentences for training (None = all)
        """
        print(f"Training n-gram model from: {srwac_path}")
        start = time.time()

        prev_word = '<S>'  # sentence start marker
        current_sid = None
        sent_count = 0

        open_func = gzip.open if srwac_path.endswith('.gz') else open

        with open_func(srwac_path, 'rt', encoding='utf-8') as f:
            reader = csv.reader(f)
            # Skip header if present
            first_row = next(reader)
            if first_row[0] != 'Token':
                # Not a header, process it
                token = first_row[0].lower()
                self.unigram_counts[token] += 1
                self.bigram_counts[prev_word][token] += 1
                prev_word = token
                self.total_tokens += 1

            for row in reader:
                if len(row) < 4:
                    continue

                token = row[0].lower()
                sid = row[3]

                # New sentence
                if sid != current_sid:
                    if current_sid is not None:
                        sent_count += 1
                        if max_sentences and sent_count >= max_sentences:
                            break
                    current_sid = sid
                    prev_word = '<S>'

                # Count
                self.unigram_counts[token] += 1
                self.bigram_counts[prev_word][token] += 1
                self.total_tokens += 1
                prev_word = token

                if self.total_tokens % 5_000_000 == 0:
                    print(f"  {self.total_tokens:,} tokens, "
                          f"{sent_count:,} sentences...")

        self._loaded = True
        elapsed = time.time() - start
        print(f"  Done: {self.total_tokens:,} tokens, "
              f"{len(self.unigram_counts):,} unique, "
              f"{sum(len(v) for v in self.bigram_counts.values()):,} bigrams, "
              f"{elapsed:.1f}s")

    def score_word_in_context(self, word, prev_word):
        """
        Computes the log-probability of a word given the previous word.

        Uses Lidstone smoothing (add-1) to avoid zero probabilities.

        Args:
            word: candidate (with diacritics)
            prev_word: previous word in the sentence

        Returns:
            Log-probability P(word | prev_word)
        """
        word_lower = word.lower()
        prev_lower = prev_word.lower() if prev_word else '<S>'

        # Bigram count
        bigram_count = self.bigram_counts[prev_lower].get(word_lower, 0)

        # Unigram count of the previous word (for normalization)
        prev_count = self.unigram_counts.get(prev_lower, 0)
        if prev_lower == '<S>':
            prev_count = sum(
                self.bigram_counts['<S>'].values()
            ) if '<S>' in self.bigram_counts else 1

        vocab_size = len(self.unigram_counts)

        # Lidstone smoothing
        prob = (bigram_count + 1) / (prev_count + vocab_size)

        return math.log(prob)

    def score_candidates(self, candidates, prev_word, next_word=None):
        """
        Scores a list of candidates in context.

        Args:
            candidates: list of candidate words
            prev_word: previous word
            next_word: next word (optional, for trigram-like scoring)

        Returns:
            Dict: {candidate: score}
        """
        scores = {}
        for word in candidates:
            score = self.score_word_in_context(word, prev_word)
            if next_word:
                score += self.score_word_in_context(next_word, word)
            scores[word] = score
        return scores

    def best_candidate(self, candidates, prev_word, next_word=None):
        """Returns the candidate with the highest score."""
        if not candidates:
            return None
        scores = self.score_candidates(candidates, prev_word, next_word)
        return max(scores, key=scores.get)

    def save(self, filepath):
        """Saves the model to disk."""
        data = {
            'bigram_counts': {k: dict(v) for k, v in self.bigram_counts.items()},
            'unigram_counts': dict(self.unigram_counts),
            'total_tokens': self.total_tokens,
        }
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"Model saved: {filepath} ({size_mb:.1f} MB)")

    def load(self, filepath):
        """Loads the model from disk."""
        print(f"Loading n-gram model: {filepath}")
        start = time.time()
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        self.bigram_counts = defaultdict(Counter)
        for k, v in data['bigram_counts'].items():
            self.bigram_counts[k] = Counter(v)
        self.unigram_counts = Counter(data['unigram_counts'])
        self.total_tokens = data['total_tokens']
        self._loaded = True
        elapsed = time.time() - start
        print(f"  Loaded in {elapsed:.1f}s")


if __name__ == '__main__':
    import sys

    srwac_path = os.path.join(
        os.path.dirname(__file__), '..', 'data',
        'srWaC1.1.01_pronouns_classified.csv'
    )

    model = NgramModel()

    # Train on first 500K sentences (fast, ~5M tokens)
    model.train_from_srwac(srwac_path, max_sentences=500_000)

    # Save
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ngram_model.json.gz')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)

    # Test
    print("\nContextual disambiguation test:")
    test_cases = [
        (['reči', 'reći', 'reci'], 'vrste', None),
        (['neće', 'neče'], 'on', None),
        (['što', 'sto'], 'sve', None),
        (['više', 'vise'], 'mnogo', None),
        (['nas', 'naš'], 'to', 'je'),
        (['Nišu', 'nisu'], 'u', None),
    ]

    for candidates, prev, next_w in test_cases:
        scores = model.score_candidates(candidates, prev, next_w)
        best = max(scores, key=scores.get)
        ctx = f"'{prev} ___'" if not next_w else f"'{prev} ___ {next_w}'"
        print(f"  {ctx:<25} {candidates} -> {best}")
