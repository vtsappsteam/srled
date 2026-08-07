#!/usr/bin/env python3
"""Preprocess v2p: itertext parsiranje (pune rečenice, kao preprocess_v2) + Pn/Pa
klasifikacija zamenica u OBA kraka. Redosled kao originalni lanac (april+jun):
prvo 8 gramatičkih pravila (samo corrected krak), PA classify_pronoun po lemi
(oba kraka) - bitno jer pravila proizvode nove P tagove (C→P) koji tek onda
dobijaju n/a podvrstu.

Klasifikacija se NE reimplementira: uvozi se doslovno classify_pronoun() iz
classify_pronouns.py (jun 2026), pa je identitet sa v3 linijom po konstrukciji.

Tagovi ostaju MEŠOVITIH SLOVA (MULTEXT-East); NLTK TaggedCorpusReader se u ovoj
liniji NE koristi (on bi ih uppercase-ovao).

Ugrađene provere:
  - identity brojača pravila (kao preprocess_v2)
  - posle upisa: poređenje sa srWaC_v2_*.txt - skidanjem n/a iz Pn/Pa tagova
    mora se dobiti TAČNO stari fajl (dokaz da je jedina razlika klasifikacija)

Izlaz: data/processed/srWaC_v2p_original.txt i srWaC_v2p_corrected.txt
Statistika: Paper 4 - Grammar Corrections Impact/data/preprocess_v2p_stats.json
"""
import sys, json, time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
from grammar_rules import apply_rules_to_sentence
from classify_pronouns import classify_pronoun
from preprocess_srwac import cyrillic_to_latin
from config import CONVERT_CYRILLIC_TO_LATIN as CONVERT

XML = BASE / 'data' / 'raw' / 'srWaC1.1.01.xml'
OUT_DIR = BASE / 'data' / 'processed'
OUT_ORIG = OUT_DIR / 'srWaC_v2p_original.txt'
OUT_CORR = OUT_DIR / 'srWaC_v2p_corrected.txt'
OLD_ORIG = OUT_DIR / 'srWaC_v2_original.txt'
OLD_CORR = OUT_DIR / 'srWaC_v2_corrected.txt'
OUT_STATS = BASE / 'data' / 'preprocess_v2p_stats.json'


def classify_arm(toks):
    """Pn/Pa klasifikacija po lemi; vraća (novi_toks, br_pn, br_pa)."""
    out = []
    pn = pa = 0
    for w, l, m in toks:
        if m.startswith('P'):
            nm = classify_pronoun(m, l)
            if nm != m:
                if nm[1] == 'n':
                    pn += 1
                elif nm[1] == 'a':
                    pa += 1
            out.append((w, l, nm))
        else:
            out.append((w, l, m))
    return out, pn, pa


def strip_pnpa(tag):
    """Pn.../Pa... -> P... (inverz klasifikacije za identity check)."""
    if len(tag) >= 2 and tag[0] == 'P' and tag[1] in 'na':
        return 'P' + tag[2:]
    return tag


def main():
    t0 = time.time()
    n_sent = n_tok = 0
    pn_o = pa_o = pn_c = pa_c = 0
    tags_orig = set()
    tags_corr = set()
    fo = open(OUT_ORIG, 'w', encoding='utf-8')
    fc = open(OUT_CORR, 'w', encoding='utf-8')
    for event, elem in ET.iterparse(str(XML), events=('end',)):
        if elem.tag == 's':
            full = ''.join(elem.itertext())
            toks = []
            for line in full.strip().split('\n'):
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    token, lemma, msd = parts[0], parts[1], parts[3]
                    if CONVERT:
                        token = cyrillic_to_latin(token)
                        lemma = cyrillic_to_latin(lemma)
                    toks.append((token, lemma, msd))
            if toks:
                n_sent += 1
                n_tok += len(toks)
                corr = apply_rules_to_sentence(toks)
                orig_p, a, b = classify_arm(toks)
                pn_o += a; pa_o += b
                corr_p, a, b = classify_arm(corr)
                pn_c += a; pa_c += b
                fo.write(' '.join(f'{w}/{m}' for w, _, m in orig_p) + '\n')
                fc.write(' '.join(f'{w}/{m}' for w, _, m in corr_p) + '\n')
                for _, _, m in orig_p: tags_orig.add(m)
                for _, _, m in corr_p: tags_corr.add(m)
            elem.clear()
            if n_sent % 500000 == 0:
                print(f'{n_sent:,} rečenica, {n_tok:,} tokena, {time.time()-t0:.0f}s', flush=True)
    fo.close(); fc.close()

    # Identity vs stari v2 fajlovi: strip(n/a) mora dati tačno stari sadržaj
    print('Identity check vs srWaC_v2_*.txt ...', flush=True)
    mism = {'original': 0, 'corrected': 0}
    for key, new_f, old_f in [('original', OUT_ORIG, OLD_ORIG),
                              ('corrected', OUT_CORR, OLD_CORR)]:
        with open(new_f, encoding='utf-8') as fn, open(old_f, encoding='utf-8') as fol:
            for ln, (a, b) in enumerate(zip(fn, fol), 1):
                stripped = ' '.join(
                    f"{t.rsplit('/', 1)[0]}/{strip_pnpa(t.rsplit('/', 1)[1])}"
                    for t in a.split())
                if stripped != b.rstrip('\n'):
                    mism[key] += 1
                    if mism[key] <= 3:
                        print(f'  MISMATCH {key} linija {ln}')

    res = {'sentences': n_sent, 'tokens': n_tok,
           'pn_original': pn_o, 'pa_original': pa_o,
           'pn_corrected': pn_c, 'pa_corrected': pa_c,
           'tags_original_case_sensitive': len(tags_orig),
           'tags_corrected_case_sensitive': len(tags_corr),
           'strip_identity_mismatches': mism,
           'runtime_s': round(time.time() - t0, 1)}
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT_STATS, 'w'), indent=1)
    print(json.dumps(res, indent=1))
    ok = mism['original'] == 0 and mism['corrected'] == 0
    print('IDENTITY:', 'PASS' if ok else 'FAIL')


if __name__ == '__main__':
    main()
