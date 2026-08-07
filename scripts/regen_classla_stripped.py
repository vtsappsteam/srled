#!/usr/bin/env python3
"""Regeneracija /tmp/classla_stripped_out.txt (obrisan sa /tmp).

Poziva IDENTIČNU run_classla_stripped funkciju iz experiment_stripped_v6_full
(subprocess u /tmp/classla_env). Posle regeneracije uporedi CLASSLA lemme sa
uskladištenim starim rezultatom (stripped_reldi_v6_full.json D=93.90 na celom
korpusu) da se validira da classla 2.2.1 reprodukuje stare brojeve.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import experiment_stripped_v6_full as ex

reldi = ex.load_reldi(ex.RELDI)
n_tok = sum(len(s) for s in reldi)
print(f'ReLDI: {len(reldi)} rečenica, {n_tok:,} tokena; pokrećem CLASSLA (stripped)...', flush=True)
lemmas = ex.run_classla_stripped(reldi)
print(f'CLASSLA gotov: {sum(len(l) for l in lemmas):,} lema u /tmp/classla_stripped_out.txt', flush=True)

# validacija: tačnost CLASSLA lema na celom korpusu prema gold lemama (D uslov)
corr = tot = 0
for sent, lsent in zip(reldi, lemmas):
    for (w, gl, _), pl in zip(sent, lsent):
        tot += 1
        corr += (pl.lower() == gl.lower())
acc = 100 * corr / tot
print(f'D uslov (ceo korpus): {acc:.2f}%  (stara referenca: {ex.LEGACY_REF["D"]})')
print('VALIDACIJA:', 'PASS' if abs(acc - ex.LEGACY_REF['D']) < 0.05 else 'RAZLIKA - proveriti verziju!')
