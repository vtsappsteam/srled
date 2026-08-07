#!/usr/bin/env python3
"""
Identity verification of the optimized implementation against the reference
implementation on ALL evaluation data used in the paper:
UD Serbian-SET (test + dev), ReLDI-NormTagNER-sr (test), SrpKor4Tagging (test).

Verifies both components:
  1. FastPerceptronTagger vs NLTK PerceptronTagger (tag sequences)
  2. CompactLemmatizer vs in-memory reference lemmatizer (lemmas, gold MSD)

"""

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from ablation_study import AblationLemmatizer, load_conllu
from compact_lemmatizer import CompactLemmatizer
from fast_tagger import FastPerceptronTagger

BASE = Path(__file__).resolve().parent.parent.parent
if not (BASE / 'POS-Aware-Stemmer').exists() and (BASE.parent / 'POS-Aware-Stemmer').exists():
    BASE = BASE.parent  # repo/scripts layout is one level deeper
SRLEX = str(BASE / 'POS-Aware-Stemmer' / 'data' / 'srLex_v1.3.gz')
EXPANDED = str(Path(__file__).parent.parent / 'data' / 'expanded_supplement_v2.json')
NLTK_MODEL = BASE / 'NLTK Treniranje' / 'models_v7_pnpa' / 'perceptron-tagger-srwac-pnpa.pickle'
FAST_DIR = BASE / 'NLTK Treniranje' / 'models_v8_fast_pnpa'
LEXICON_DIR = BASE / 'Diacritics-Restoration' / 'data' / 'compact_lexicon'
RELDI = BASE / 'data' / 'ReLDI-NormTagNER-sr' / 'reldi-normtagner-sr.conllup'

UPOS_TO_MSD = {
    'NOUN': 'N', 'PROPN': 'N', 'VERB': 'V', 'AUX': 'V', 'ADJ': 'A',
    'PRON': 'P', 'DET': 'P', 'ADV': 'R', 'ADP': 'S', 'CCONJ': 'C',
    'SCONJ': 'C', 'NUM': 'M', 'PART': 'Q', 'INTJ': 'I', 'X': 'X', 'SYM': 'X',
}


def load_reldi_test(path):
    test, current, is_test = [], [], False
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current and is_test:
                    test.append(current)
                current = []
                continue
            if line.startswith('# contained_in_datasets'):
                is_test = 'test' in line
                continue
            if line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 5 and '-' not in parts[0] and '.' not in parts[0]:
                if parts[2] != '_':
                    current.append((parts[1], parts[2], parts[4]))
    if current and is_test:
        test.append(current)
    return test


def load_srpkor_test():
    from datasets import load_dataset
    ds = load_dataset('jerteh/SrpKor4Tagging')['train']
    test_size = len(ds) // 10
    sentences = []
    for idx in range(len(ds) - test_size, len(ds)):
        sent = []
        for tok, lem, upos in zip(ds['token'][idx], ds['lemma'][idx], ds['ud'][idx]):
            if upos != 'PUNCT':
                sent.append((tok, lem, UPOS_TO_MSD.get(upos, 'X')))
        if sent:
            sentences.append(sent)
    return sentences


def main():
    testsets = {}
    testsets['UD test'] = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-test.conllu')
    testsets['UD dev'] = load_conllu('/tmp/UD_Serbian-SET/sr_set-ud-dev.conllu')
    testsets['ReLDI test'] = load_reldi_test(RELDI)
    try:
        testsets['SrpKor test'] = load_srpkor_test()
    except Exception as e:
        print(f"WARNING: SrpKor unavailable ({e}) - verification will exclude it", flush=True)

    for name, s in testsets.items():
        print(f"{name}: {len(s)} sentences, {sum(len(x) for x in s):,} tokens", flush=True)
    grand_total = sum(sum(len(x) for x in s) for s in testsets.values())
    print(f"TOTAL: {grand_total:,} tokens\n", flush=True)

    # ---- 1. Lemmatizer identity ----
    print("Loading reference lemmatizer...", flush=True)
    ref_lem = AblationLemmatizer(SRLEX, expanded_dict_path=EXPANDED)
    compact = CompactLemmatizer(LEXICON_DIR)

    lem_total = lem_mism = 0
    for name, sents in testsets.items():
        n = m = 0
        for sent in sents:
            for w, gl, msd in sent:
                a = ref_lem.lemmatize(w, msd)
                b = compact.lemmatize(w, msd)
                n += 1
                if a != b:
                    m += 1
                    if lem_mism + m <= 5:
                        print(f"  LEM MISMATCH [{name}]: {w!r}/{msd} ref={a!r} compact={b!r}", flush=True)
        print(f"  Lemmatizer {name}: {n - m}/{n} identical", flush=True)
        lem_total += n
        lem_mism += m
    print(f"LEMMATIZER identity: {lem_total - lem_mism}/{lem_total} "
          f"({'PASS' if lem_mism == 0 else 'FAIL'})\n", flush=True)
    del ref_lem

    # ---- 2. Tagger identity ----
    print("Loading NLTK tagger (reference)...", flush=True)
    with open(NLTK_MODEL, 'rb') as f:
        nltk_tagger = pickle.load(f)
    fast = FastPerceptronTagger.load(FAST_DIR)

    tag_total = tag_mism = 0
    for name, sents in testsets.items():
        n = m = 0
        t0 = time.time()
        for sent in sents:
            words = [w for w, gl, msd in sent]
            ref_tags = nltk_tagger.tag(words)
            fast_tags = fast.tag(words)
            for (w1, t1), (w2, t2) in zip(ref_tags, fast_tags):
                n += 1
                if t1 != t2:
                    m += 1
                    if tag_mism + m <= 5:
                        print(f"  TAG MISMATCH [{name}]: {w1!r} nltk={t1} fast={t2}", flush=True)
        print(f"  Tagger {name}: {n - m}/{n} identical ({time.time()-t0:.0f}s)", flush=True)
        tag_total += n
        tag_mism += m
    print(f"TAGGER identity: {tag_total - tag_mism}/{tag_total} "
          f"({'PASS' if tag_mism == 0 else 'FAIL'})", flush=True)


if __name__ == '__main__':
    main()
