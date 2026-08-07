# SrLeD: POS-Aware Pipeline for Diacritics Restoration and Lemmatization of Serbian

Source code and evaluation scripts accompanying the paper:

> **A Resource-Efficient POS-Aware Pipeline for Diacritics Restoration and Lemmatization of Serbian**
> Nikola Vukotić and Suzana Stojković
> Under journal review, 2026

## Overview

This repository contains the implementation of SrLeD (Serbian Lemmatization and Diacritics), an integrated NLP pipeline for Serbian that performs diacritics restoration and lemmatization within a unified architecture. The system uses a cascade lemmatization strategy with four resolution layers preceded by a diacritics restoration preprocessing step, all guided by POS tags from an NLTK averaged perceptron tagger and the srLex v1.3 morphological dictionary.

**Key results:**
- Lemmatization with gold tags: 98.48% on UD-SET, 94.43% on ReLDI Twitter, 88.95% on SrpKor4Tagging; equivalent to CLASSLA-Stanza within 0.5 pp on the first two domains (TOST) and significantly more accurate on the third (*p* < 0.001)
- Lemmatization end-to-end (own predicted MSD tags): 98.18% on UD-SET, 93.54% on ReLDI, 88.38% on SrpKor4Tagging
- Diacritics restoration: 99.56% word accuracy on SETimes.SR (above redi without its 7.6 GB language model)
- Efficiency (Apple M2, CPU only, scenario-based, optimized implementation with verified-identical outputs on all 61,778 evaluation tokens): 28x higher batch throughput than CLASSLA-Stanza (180K vs 6.5K tok/s), 40x lower single-sentence latency (0.54 vs 21.4 ms), 10x less peak memory (0.45 vs 4.44 GB), near-instant model loading (0.06 s), no GPU required.

## Repository structure

```
repo/
  src/                          Core library modules
    diacritics.py               Diacritical character definitions and utilities
    candidate_generator.py      srLex-based candidate generation (+ expanded supplement)
    pos_disambiguator.py        POS-guided disambiguation (dominance guard, Np handling)
    v6_restorer.py              Restoration module facade used as lemmatizer Layer 0
    ngram_model.py              Bigram/trigram language model (development experiment)
    oov_handler.py              Out-of-vocabulary word handling (suffix, dz/dj rules)
    evaluator.py                Evaluation framework
    fast_tagger.py              Vectorized memory-mapped perceptron tagger
    compact_lemmatizer.py       Memory-mapped trie-based lexicon backend
    compact_v6.py               Memory-mapped backend for the restoration module
                                (shares the decision logic with the reference)
  scripts/                      Evaluation and analysis scripts
    pos_lemmatizer_v2.py        Cascade lemmatizer (main system)
    finetune_ud_clean.py        POS tagger fine-tuning (zero data leakage)
    expand_dictionary.py        Build the expanded supplementary dictionary
    evaluate_on_splits.py       Lemmatization evaluation on UD test split
    evaluate_three_testsets.py  Cross-domain evaluation (UD, ReLDI, SrpKor4Tagging)
    evaluate_end_to_end.py      Gold vs predicted MSD evaluation
    experiment_stripped_reldi.py Robustness evaluation on stripped text
    ablation_study.py           Ablation study
    statistical_tests.py        McNemar tests for per-POS and ablation tables
    run_evaluation.py           Diacritics restoration evaluation
    reeval_diacritics_v6.py     Diacritics evaluation with srWaC tables + dominance guard
    mine_srwac_supplement.py    Mine srWaC for the supplementary and proper-noun tables
    guard_sensitivity.py        Sensitivity analysis of the frequency dominance guard
    diagnose_real_errors.py     Categorized error analysis of the production system
    analyze_diacritics_errors.py Diacritics error analysis
    experiment_stripped_v6_full.py Stripped-ReLDI comparison incl. CLASSLA + McNemar (Table 4)
    granularity_full_system.py  MSD granularity experiment (Table 5)
    lemma_error_categories.py   Lemmatization error categorization (Section 5.8)
    benchmark_classla_scenarios.py Scenario benchmark for CLASSLA (batch + single-stream)
    benchmark_efficiency.py     Efficiency benchmark for the baseline implementation
    benchmark_optimized.py      Efficiency benchmark for the optimized implementation
    benchmark_classla.py        Efficiency benchmark for CLASSLA-Stanza
    build_fast_tagger.py        Convert NLTK tagger to compact format (+ identity check)
    build_compact_lexicon.py    Build memory-mapped lexicon (+ identity check)
    verify_identity_all.py      Identity verification on all evaluation data (3 domains)
    final_system_eval.py        Aggregated accuracy evaluation (single entry point)
    scispace_experiments.py     End-to-end + CLASSLA nonstandard comparisons, TOST,
                                McNemar (writes per-token vectors, scispace_pertoken.npz)
    stripped_test_split_eval.py Stripped ReLDI test split, gold-MSD condition
    stripped_predicted_msd_eval.py Stripped ReLDI test split, predicted-MSD condition
    setimes_testsplit_restoration.py Leakage check harness (held-out SETimes split)
    build_supplement_v2.py      Build the v2 expanded supplementary dictionary
    mine_lemma_corrections.py   Mine lemma correction candidates from gold corpora
    reviewed_corrections_eval.py Evaluate the reviewed 148-entry correction set
    dict_improvements_eval.py   Dictionary improvements evaluation (Section 5 discussion)
    pnpa_2x2_experiment.py      Pretraining vs pronoun-subclassification control (2x2)
    gold_only_experiment.py     Gold-corpora-only tagger control experiment

    The *_v2p.py and *_rerun.py scripts re-run the corresponding evaluations
    with the retrained tagger (the deployed system reported in the paper).
    Each redirects only the tagger/model path and writes to a new output file;
    tagger-independent (gold-tag) results serve as built-in identity controls.

    final_system_eval_v2p.py    Full accuracy re-run (Tables 2, 5, 6 and Section 5)
    run_stripped_v2p.py         Stripped-ReLDI re-run: gold-MSD control + predicted MSD
    stripped_pred_vs_nonstandard_v2p.py Full pipeline vs CLASSLA nonstandard (Table 4)
    reeval_diacritics_v2p.py    Diacritics restoration re-run (Table 3)
    restoration_error_split_v2p.py Table 3 error counters, false changes, raw-token share
    diagnose_real_errors_v2p.py Restoration error categories (Table 7)
    restoration_pos_ablation_v2p.py POS ablation of restoration on news (paired McNemar)
    guard_sensitivity_v2p.py    Dominance-guard ablation + threshold sensitivity
    setimes_testsplit_restoration_v2p.py Leakage check on the held-out SETimes test split
    udonly_variant_eval_v2p.py  Split-clean UD-only tagger variant evaluation
    tagger_degradation_heldout_v2p.py Tagger robustness to stripped input (held-out)
    regen_classla_stripped.py   Regenerate CLASSLA stripped output (+ 2.2.1 validation)
    benchmark_optimized_v2p.py  Efficiency re-run for the optimized implementation
    benchmark_classla_rerun.py  Efficiency re-run for CLASSLA-Stanza
    benchmark_classla_scenarios_rerun.py Scenario benchmark re-run for CLASSLA
    benchmark_efficiency_v2p.py Naive-implementation and hash-backend benchmarks
    build_fast_tagger_v2p.py    Compact build of the retrained tagger (+ identity check)
    verify_identity_all_v2p.py  Identity verification of the optimized retrained system
    generate_fig_system_overview_v2p.py Programmatic Fig. 1 variant (numbers from JSONs)
    generate_fig_tradeoff_v2p.py Fig. 4 generator (reads benchmark JSONs)
    generate_fig_tradeoff_v2p_a11y.py Fig. 4 as submitted (color-blind/grayscale safe)
    srwac_preprocessing/        Full-sentence srWaC extraction that produced the
                                tagger pre-training data (Section 4.2)
      preprocess_v2p.py         Parser (itertext, full sentences) + both corpus arms
      grammar_rules.py          The eight grammar-motivated annotation rules
      classify_pronouns.py      Pn/Pa pronoun subclassification (by lemma)
      preprocess_srwac.py       Base corpus reader (Cyrillic-to-Latin, chunking)
      config.py                 Paths and MULTEXT-East configuration
  data/                         Supplementary data
    expanded_supplement_v2.json Expanded dictionary (9,501 forms, 10,217 entries)
    expanded_supplement.json    Earlier 7,472-form version (kept for the incremental
                                development history, Step 5 in Section 5.7)
    srwac_supplement_v2.json    srWaC-derived OOV diacritics dictionary (24,114 forms)
    srwac_augment_np.json       srWaC-derived proper-noun restoration table (26,609 forms)
    ekavization_mappings.csv    17 ijekavian-to-ekavian lemma normalization rules
    lemma_corrections_v2.csv    148 dictionary correction mappings (17 manual + 131
                                mined and individually reviewed)
    lemma_corrections.csv       Initial 17 manual correction mappings
  results/                      Result JSONs behind every number in the paper, run
                                logs, and the figures as submitted (results/figures/)
  requirements.txt              Python dependencies
```

## Requirements

- Python 3.10+
- NLTK, marisa-trie
- NumPy, SciPy, Matplotlib

```bash
pip install -r requirements.txt
```

## External data (not included)

The following resources are required but not distributed with this repository due to licensing:

- **srLex v1.3** morphological dictionary: available from [CLARIN.SI](https://www.clarin.si/)
- **SETimes.SR** corpus: available from CLARIN.SI under CC BY-SA 4.0
- **ReLDI-NormTagNER-sr** corpus: available from CLARIN.SI under CC BY-SA 4.0
- **UD Serbian-SET** treebank: available from [Universal Dependencies](https://universaldependencies.org/)

## Reproducing results

The numbers reported in the paper come from the `*_v2p.py` / `*_rerun.py`
scripts (retrained tagger, the deployed system). The corresponding result
files are included in `results/`.

### 1. Full accuracy evaluation (Tables 2, 5, 6; Section 5)
```bash
python scripts/final_system_eval_v2p.py
```

### 2. Stripped-diacritics robustness (Table 4)
```bash
python scripts/run_stripped_v2p.py
python scripts/stripped_pred_vs_nonstandard_v2p.py
```

### 3. Diacritics restoration (Tables 3, 7; Section 5.4, 5.9)
```bash
python scripts/reeval_diacritics_v2p.py
python scripts/restoration_error_split_v2p.py
python scripts/diagnose_real_errors_v2p.py
python scripts/guard_sensitivity_v2p.py
python scripts/restoration_pos_ablation_v2p.py
python scripts/setimes_testsplit_restoration_v2p.py
```

### 4. Tagger variants and robustness (Sections 4.2, 5.3)
```bash
python scripts/udonly_variant_eval_v2p.py
python scripts/tagger_degradation_heldout_v2p.py
```

### 5. Build the optimized implementation and verify identity (Section 4.6)
```bash
python scripts/build_fast_tagger_v2p.py
python scripts/build_compact_lexicon.py
python scripts/verify_identity_all_v2p.py
```

### 6. Efficiency benchmarks (Table 8, Figure 4)
```bash
python scripts/benchmark_optimized_v2p.py
python scripts/benchmark_classla_scenarios_rerun.py
python scripts/benchmark_classla_rerun.py
python scripts/benchmark_efficiency_v2p.py
```

### 7. Figures
```bash
python scripts/generate_fig_system_overview_v2p.py
python scripts/generate_fig_tradeoff_v2p_a11y.py
```

The development-line scripts (`finetune_ud_clean.py`, `expand_dictionary.py`,
`evaluate_three_testsets.py`, `ablation_study.py`, `statistical_tests.py`,
`run_evaluation.py`, ...) are the modules the re-run wrappers import and
document the original construction of the system; gold-tag results are
tagger-independent and identical between the two lines.

## Supplementary micro-resource

The `data/` directory contains supplementary resources:

- **`expanded_supplement_v2.json`**: 9,501 forms with 10,217 word-lemma-MSD entries extracted from gold-annotated corpora (SETimes 2.0 + ReLDI train splits), covering OOV words not in srLex; `expanded_supplement.json` is the earlier 7,472-form version (development Step 5)
- **`srwac_supplement_v2.json`**: 24,114 ASCII-form → diacritized-form entries mined from a 100M-token portion of srWaC (frequency ≥ 3, diacritized variant at least twice as frequent as the ASCII spelling); used for OOV diacritics restoration
- **`srwac_augment_np.json`**: 26,609 capitalized proper-noun entries mined case-sensitively from srWaC with the same dominance criterion; consulted when the tagger assigns Np and srLex offers no attested proper-noun reading
- **`ekavization_mappings.csv`**: 17 ijekavian-to-ekavian lemma normalization rules
- **`lemma_corrections_v2.csv`**: 148 dictionary correction mappings addressing known issues in srLex (17 manual + 131 mined from gold corpora and individually reviewed; the review record is `results/lemma_correction_candidates_reviewed.csv`); `lemma_corrections.csv` is the initial 17-mapping version

These resources are extracted and verified from the system described in the paper and may be useful for other Serbian NLP tools.

## License

MIT License

## Citation

The paper is under review; a citation entry will be added upon publication.
