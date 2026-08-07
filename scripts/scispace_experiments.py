#!/usr/bin/env python3
"""
Dopunski eksperimenti posle interne verifikacije recenzije (16. jul 2026):

  E1. End-to-end (predicted MSD) lematizacija na SVA TRI test seta,
      uključujući SrpKor4Tagging (ranije nemereno: tager daje pun MSD
      nezavisno od toga što je gold anotacija UPOS).
  E2. McNemar test: naš sistem (predicted MSD) vs CLASSLA, po domenu.
  E3. CLASSLA nonstandard varijanta na ReLDI (fer poređenje za tvitove:
      nonstandard je trenirana na ReLDI trening skupu, kao i naš tager).
  E4. 95% CI za uparenu razliku tačnosti + TOST ekvivalencija (±0.5 pp)
      za gold-vs-CLASSLA i pred-vs-CLASSLA parove.

Ponovo računa i postojeće brojeve (gold, pred-UD, CLASSLA standard) kao
sanity check protiv three_testsets_evaluation.json.

Upotreba (dugotrajno - pokretati detached):
    nohup python3 scripts/scispace_experiments.py > results/scispace_experiments.log 2>&1 &
Izlaz: results/scispace_experiments.json (+ per-token vektori u .npz)
"""

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
import evaluate_three_testsets as base  # noqa: E402

# repo/scripts je jedan nivo dublje od originala - ispravi BASE-zavisne putanje
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # .../NLP-POS-Tagging
base.BASE = PROJECT_ROOT
base.SRLEX = str(PROJECT_ROOT / "POS-Aware-Stemmer" / "data" / "srLex_v1.3.gz")
base.V7_MODEL = str(PROJECT_ROOT / "NLTK Treniranje" / "models_v7" / "perceptron-tagger-expanded.pickle")

OUT_JSON = SCRIPT_DIR.parent / "results" / "scispace_experiments.json"
OUT_NPZ = SCRIPT_DIR.parent / "results" / "scispace_pertoken.npz"
TOST_MARGIN = 0.005  # ±0.5 pp


def run_classla_variant(test_sents, name, classla_type=None):
    """Kao base.run_classla, uz opcioni type (npr. 'nonstandard')."""
    tmp_input = f"/tmp/classla_{name}.conllu"
    with open(tmp_input, "w", encoding="utf-8") as f:
        for i, sent in enumerate(test_sents):
            f.write(f"# sent_id = {i}\n")
            for j, (w, l, m) in enumerate(sent, 1):
                f.write(f"{j}\t{w}\t_\t_\t_\t_\t_\t_\t_\t_\n")
            f.write("\n")
    type_arg = f', type="{classla_type}"' if classla_type else ""
    dl_type = f', type="{classla_type}"' if classla_type else ""
    script = f'''
import classla
try:
    classla.download("sr"{dl_type}, verbose=False)
except Exception:
    pass
nlp = classla.Pipeline("sr", processors="tokenize,pos,lemma",
                        tokenize_pretokenized=True, use_gpu=False{type_arg})
sentences = []
current = []
with open("{tmp_input}") as f:
    for line in f:
        line = line.strip()
        if not line:
            if current: sentences.append(current); current = []
            continue
        if line.startswith("#"): continue
        parts = line.split("\\t")
        if len(parts) >= 2: current.append(parts[1])
if current: sentences.append(current)
doc = nlp([[w for w in sent] for sent in sentences])
for sent in doc.sentences:
    print("\\t".join([w.lemma if w.lemma else "" for w in sent.words]))
'''
    tmp_script = f"/tmp/classla_{name}.py"
    with open(tmp_script, "w") as f:
        f.write(script)
    r = subprocess.run([base.CLASSLA_PY, tmp_script],
                       capture_output=True, text=True, timeout=3600)
    if r.returncode != 0:
        print(f"  CLASSLA GREŠKA ({name}): {r.stderr[-400:]}", flush=True)
        return None
    return [line.split("\t") for line in r.stdout.strip().split("\n") if line.strip()]


def correctness_vec_classla(test_sents, cl_lemmas):
    v = []
    for i, sent in enumerate(test_sents):
        for j, (w, gl, msd) in enumerate(sent):
            ok = 0
            if i < len(cl_lemmas) and j < len(cl_lemmas[i]):
                cl = cl_lemmas[i][j]
                ok = 1 if cl and gl and cl.lower() == gl.lower() else 0
            v.append(ok)
    return np.array(v, dtype=np.int8)


def mcnemar_exact(a, b):
    """Egzaktni binomni McNemar (uz hi-kvadrat za poređenje sa starim)."""
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    if n01 + n10 == 0:
        return {"p_exact": 1.0, "p_chi2": 1.0, "only_ours": n01, "only_classla": n10}
    p_exact = stats.binomtest(min(n01, n10), n01 + n10, 0.5).pvalue * 1  # two-sided
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return {"p_exact": round(float(p_exact), 4),
            "p_chi2": round(float(1 - stats.chi2.cdf(chi2, 1)), 4),
            "only_ours": n01, "only_classla": n10}


def paired_stats(a, b):
    """95% CI uparene razlike (a-b) + TOST ekvivalencija na ±0.5 pp."""
    d = a.astype(float) - b.astype(float)
    n = len(d)
    mean = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    ci = (mean - 1.96 * se, mean + 1.96 * se)
    # TOST: H0a: mean <= -margin; H0b: mean >= +margin
    t_low = (mean + TOST_MARGIN) / se
    t_high = (mean - TOST_MARGIN) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)      # test protiv donje granice
    p_high = stats.t.cdf(t_high, n - 1)        # test protiv gornje granice
    p_tost = max(p_low, p_high)
    return {"diff_pp": round(mean * 100, 3),
            "ci95_pp": [round(ci[0] * 100, 3), round(ci[1] * 100, 3)],
            "tost_margin_pp": TOST_MARGIN * 100,
            "tost_p": round(float(p_tost), 5),
            "equivalent_at_margin": bool(p_tost < 0.05)}


def main():
    print("Učitavanje lematizatora i tagera...", flush=True)
    lem = base.Lemmatizer(base.SRLEX, base.EXTRA)
    with open(base.V7_MODEL, "rb") as f:
        tagger = pickle.load(f)

    testsets = {}
    testsets["UD-SET (news)"] = base.load_conllu("/tmp/UD_Serbian-SET/sr_set-ud-test.conllu")
    testsets["ReLDI (Twitter)"] = base.load_reldi_test(
        str(base.BASE / "data" / "ReLDI-NormTagNER-sr" / "reldi-normtagner-sr.conllup"))
    testsets["SrpKor (lit+admin)"] = base.load_srpkor_test()

    results, vectors = {}, {}
    for name, sents in testsets.items():
        tag = name.replace(" ", "_").replace("(", "").replace(")", "")
        n_tok = sum(len(s) for s in sents)
        print(f"\n=== {name}: {len(sents)} rečenica, {n_tok:,} tokena ===", flush=True)

        # naš sistem - gold tagovi (SrpKor: gold je grubi UPOS→slovo, kao u radu)
        our_gold = []
        for sent in sents:
            for w, gl, msd in sent:
                pl = lem.lemmatize(w, msd)
                our_gold.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
        our_gold = np.array(our_gold, dtype=np.int8)
        print(f"  gold tagovi:      {100*our_gold.mean():.2f}%", flush=True)

        # naš sistem - PREDIKTOVANI MSD (uklj. SrpKor - E1)
        pred_tags = tagger.tag_sents([[w for w, l, m in s] for s in sents])
        our_pred = []
        for si, sent in enumerate(sents):
            for ti, (w, gl, msd) in enumerate(sent):
                pl = lem.lemmatize(w, pred_tags[si][ti][1])
                our_pred.append(1 if pl and gl and pl.lower() == gl.lower() else 0)
        our_pred = np.array(our_pred, dtype=np.int8)
        print(f"  predicted MSD:    {100*our_pred.mean():.2f}%", flush=True)

        # CLASSLA standard
        print("  CLASSLA standard...", flush=True)
        cl_std = run_classla_variant(sents, tag)
        cl_std_v = correctness_vec_classla(sents, cl_std) if cl_std else None
        if cl_std_v is not None:
            print(f"  CLASSLA standard: {100*cl_std_v.mean():.2f}%", flush=True)

        entry = {"tokens": n_tok,
                 "our_gold": round(100 * float(our_gold.mean()), 2),
                 "our_pred": round(100 * float(our_pred.mean()), 2)}
        vectors[f"{tag}_our_gold"] = our_gold
        vectors[f"{tag}_our_pred"] = our_pred

        if cl_std_v is not None:
            entry["classla_standard"] = round(100 * float(cl_std_v.mean()), 2)
            entry["gold_vs_classla"] = {**mcnemar_exact(our_gold, cl_std_v),
                                        **paired_stats(our_gold, cl_std_v)}
            entry["pred_vs_classla"] = {**mcnemar_exact(our_pred, cl_std_v),
                                        **paired_stats(our_pred, cl_std_v)}
            vectors[f"{tag}_classla_std"] = cl_std_v

        # E3: CLASSLA nonstandard SAMO za ReLDI
        if "ReLDI" in name:
            print("  CLASSLA nonstandard...", flush=True)
            cl_ns = run_classla_variant(sents, tag + "_ns", classla_type="nonstandard")
            if cl_ns:
                cl_ns_v = correctness_vec_classla(sents, cl_ns)
                entry["classla_nonstandard"] = round(100 * float(cl_ns_v.mean()), 2)
                entry["gold_vs_classla_ns"] = {**mcnemar_exact(our_gold, cl_ns_v),
                                               **paired_stats(our_gold, cl_ns_v)}
                entry["pred_vs_classla_ns"] = {**mcnemar_exact(our_pred, cl_ns_v),
                                               **paired_stats(our_pred, cl_ns_v)}
                vectors[f"{tag}_classla_ns"] = cl_ns_v
                print(f"  CLASSLA nonstd:   {100*cl_ns_v.mean():.2f}%", flush=True)

        results[name] = entry
        # checkpoint posle svakog domena
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
        np.savez_compressed(OUT_NPZ, **vectors)
        print(f"  checkpoint: {OUT_JSON}", flush=True)

    print("\nGOTOVO. Rezime:", flush=True)
    print(json.dumps(results, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
