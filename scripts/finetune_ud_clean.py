#!/usr/bin/env python3
"""
Clean fine-tuning: POS tagger fine-tuned ONLY on UD train split.

Ensures zero data leakage: UD test split sentences are never seen during training.

Steps:
  1. Load pre-trained v3 model (trained on srWaC)
  2. Fine-tune on UD Serbian-SET train split (3,328 sentences)
  3. Evaluate on UD test split (520 sentences)
  4. Save as models_v5/perceptron-tagger-ud-clean.pickle

"""

import os
import sys
import pickle
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent

# Add training module to path for imports
sys.path.insert(0, str(PROJECT_DIR / 'NLTK Treniranje'))

import nltk


def load_conllu_as_tagged(path, msd_col=4):
    """Load CoNLL-U file as NLTK tagged sentences: [[(word, tag), ...], ...]"""
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
            if len(parts) <= msd_col or '-' in parts[0] or '.' in parts[0]:
                continue
            word = parts[1]
            tag = parts[msd_col]  # XPOS = MSD tag
            current.append((word, tag))
    if current:
        sentences.append(current)
    return sentences


def evaluate(tagger, test_sents):
    """Evaluate tagger accuracy."""
    correct_pos = 0
    correct_msd = 0
    total = 0

    words_only = [[w for w, t in s] for s in test_sents]
    tagged = tagger.tag_sents(words_only)

    for gold_sent, pred_sent in zip(test_sents, tagged):
        for (_, gold_tag), (_, pred_tag) in zip(gold_sent, pred_sent):
            total += 1
            if gold_tag and pred_tag:
                if gold_tag[0].upper() == pred_tag[0].upper():
                    correct_pos += 1
                if gold_tag == pred_tag:
                    correct_msd += 1

    return {
        'pos_accuracy': round(100 * correct_pos / total, 2) if total else 0,
        'msd_accuracy': round(100 * correct_msd / total, 2) if total else 0,
        'total': total,
    }


def main():
    # Paths
    v3_model = PROJECT_DIR / 'NLTK Treniranje' / 'models_v3' / 'perceptron-tagger.pickle'
    output_dir = PROJECT_DIR / 'NLTK Treniranje' / 'models_v5'
    output_model = output_dir / 'perceptron-tagger-ud-clean.pickle'

    ud_dir = Path('/tmp/UD_Serbian-SET')
    train_path = ud_dir / 'sr_set-ud-train.conllu'
    dev_path = ud_dir / 'sr_set-ud-dev.conllu'
    test_path = ud_dir / 'sr_set-ud-test.conllu'

    # Verify files exist
    for p in [v3_model, train_path, test_path]:
        if not p.exists():
            print(f"Missing: {p}")
            sys.exit(1)

    print("=" * 60)
    print("CLEAN FINE-TUNING: UD TRAIN SPLIT ONLY (NO DATA LEAKAGE)")
    print("=" * 60)

    # Load v3 model
    print(f"\n1. Loading pre-trained v3 model: {v3_model}")
    with open(v3_model, 'rb') as f:
        tagger = pickle.load(f)
    print("   Model loaded.")

    # Load UD splits
    print(f"\n2. Loading UD Serbian-SET splits...")
    train_sents = load_conllu_as_tagged(str(train_path))
    test_sents = load_conllu_as_tagged(str(test_path))
    dev_sents = load_conllu_as_tagged(str(dev_path)) if dev_path.exists() else []

    n_train_tok = sum(len(s) for s in train_sents)
    n_test_tok = sum(len(s) for s in test_sents)
    n_dev_tok = sum(len(s) for s in dev_sents)

    print(f"   Train: {len(train_sents)} sentences, {n_train_tok:,} tokens")
    print(f"   Dev:   {len(dev_sents)} sentences, {n_dev_tok:,} tokens")
    print(f"   Test:  {len(test_sents)} sentences, {n_test_tok:,} tokens")

    # Evaluate BEFORE fine-tuning
    print(f"\n3. Evaluation BEFORE fine-tuning (v3 on UD test):")
    before = evaluate(tagger, test_sents)
    print(f"   POS accuracy: {before['pos_accuracy']}%")
    print(f"   MSD accuracy: {before['msd_accuracy']}%")
    print(f"   Total tokens: {before['total']:,}")

    # Fine-tune on UD train split ONLY
    print(f"\n4. Fine-tuning on UD train split ({len(train_sents)} sentences, 3 iterations)...")
    t0 = time.time()
    tagger.train(train_sents, nr_iter=3)
    elapsed = time.time() - t0
    print(f"   Done in {elapsed:.1f}s")

    # Evaluate AFTER fine-tuning
    print(f"\n5. Evaluation AFTER fine-tuning (v5 on UD test):")
    after = evaluate(tagger, test_sents)
    print(f"   POS accuracy: {after['pos_accuracy']}% (before: {before['pos_accuracy']}%)")
    print(f"   MSD accuracy: {after['msd_accuracy']}% (before: {before['msd_accuracy']}%)")
    print(f"   POS change: {after['pos_accuracy'] - before['pos_accuracy']:+.2f} pp")
    print(f"   MSD change: {after['msd_accuracy'] - before['msd_accuracy']:+.2f} pp")

    if dev_sents:
        print(f"\n   Dev set evaluation:")
        dev_result = evaluate(tagger, dev_sents)
        print(f"   POS accuracy: {dev_result['pos_accuracy']}%")
        print(f"   MSD accuracy: {dev_result['msd_accuracy']}%")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n6. Saving clean model: {output_model}")
    with open(output_model, 'wb') as f:
        pickle.dump(tagger, f)
    model_size = output_model.stat().st_size / (1024 * 1024)
    print(f"   Size: {model_size:.0f} MB")

    print(f"\n{'='*60}")
    print("DONE -- Model fine-tuned with ZERO data leakage.")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
