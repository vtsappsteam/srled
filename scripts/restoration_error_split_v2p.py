#!/usr/bin/env python3
"""Podela gresaka restauracije na tokene SA i BEZ dijakritike u zlatnom obliku.

Cetiri broja u radu izvode se iz agregata evaluacije, ne iz novog merenja:
  - broj gresaka i tacno restauriranih reci na SETimes (Tabela 3),
  - tacnost na tokenima koji u zlatnom obliku nose dijakritiku,
  - "false changes": greske na tokenima kojima dijakritika ne treba (mi vs redi),
  - udeo ReLDI gresaka na tokenima bez dijakritike u zlatnom obliku.

Metod je verifikovan tako sto nad STARIM JSON-om reprodukuje vrednosti koje
stoje u radu (78 false changes, 5.223 od 5.674 = 92,1%). Bez tog PASS-a
skripta ne ispisuje nove vrednosti.

Pokretanje: python3 scripts/restoration_error_split_v2p.py
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / 'results'
OLD = RES / 'evaluation_v6_improved.json'          # v5 tager (stanje u radu)
NEW = RES / 'evaluation_v2p_restoration.json'      # v2p_udonly tager
REDI = RES / 'redi_full_results.json'


def split(entry):
    """(ukupno gresaka, greske na dijakritik-nosecim, greske na ostalima)."""
    err = entry['total_words'] - entry['correct_words']
    diac_err = entry['total_diac_words'] - entry['correct_diac_words']
    return err, diac_err, err - diac_err


def pick(data, needle):
    return next(v for k, v in data.items() if needle in k)


def main():
    old = json.loads(OLD.read_text(encoding='utf-8'))
    new = json.loads(NEW.read_text(encoding='utf-8'))
    redi = json.loads(REDI.read_text(encoding='utf-8'))

    # --- identity: metod mora reprodukovati brojeve koji su vec u radu ---
    o_set = pick(old, 'POS-Aware v6_SETimes')
    o_rel = pick(old, 'POS-Aware v6_ReLDI')
    r_set = redi['redi_noLM_SETimes']
    checks = [
        ('SETimes greske (rad: 319)', split(o_set)[0], 319),
        ('SETimes tacno (rad: 74.412)', o_set['correct_words'], 74412),
        ('SETimes false changes (rad: 78)', split(o_set)[2], 78),
        ('redi noLM false changes (rad: 204)', split(r_set)[2], 204),
        ('ReLDI greske (rad: 5.674)', split(o_rel)[0], 5674),
        ('ReLDI bez dijakritike (rad: 5.223)', split(o_rel)[2], 5223),
    ]
    print('IDENTITY nad starim rezultatima:')
    fails = []
    for name, got, want in checks:
        ok = got == want
        print(f'  {"PASS" if ok else "FAIL"}  {name:<40} {got:>8,}')
        if not ok:
            fails.append(f'{name}: {got} != {want}')
    if fails:
        raise SystemExit('IDENTITY FAIL: ' + '; '.join(fails))

    # --- nove vrednosti ---
    n_set = pick(new, 'POS-Aware v6_SETimes')
    n_rel = pick(new, 'POS-Aware v6_ReLDI')
    e_set, d_set, f_set = split(n_set)
    e_rel, d_rel, f_rel = split(n_rel)
    print('\nNOVE VREDNOSTI (v2p_udonly tager):')
    print(f'  SETimes: {n_set["correct_words"]:,} tacno, {e_set} gresaka '
          f'(staro {o_set["correct_words"]:,} / {split(o_set)[0]})')
    print(f'  tacnost na dijakritik-nosecim tokenima: {n_set["diac_word_accuracy"]}% '
          f'({n_set["correct_diac_words"]:,}/{n_set["total_diac_words"]:,}; staro '
          f'{o_set["diac_word_accuracy"]}%)')
    print(f'  false changes: nas sistem {f_set} vs redi bez LM {split(r_set)[2]} '
          f'(staro {split(o_set)[2]} vs {split(r_set)[2]})')
    print(f'  ReLDI: {e_rel:,} gresaka, od toga {f_rel:,} na tokenima bez dijakritike '
          f'u zlatnom obliku = {100 * f_rel / e_rel:.1f}% '
          f'(staro {split(o_rel)[2]:,} od {split(o_rel)[0]:,} = '
          f'{100 * split(o_rel)[2] / split(o_rel)[0]:.1f}%)')

    out = RES / 'restoration_error_split_v2p.json'
    out.write_text(json.dumps({
        'setimes': {'correct_words': n_set['correct_words'], 'errors': e_set,
                    'diac_errors': d_set, 'false_changes': f_set,
                    'diac_word_accuracy': n_set['diac_word_accuracy'],
                    'redi_nolm_false_changes': split(r_set)[2]},
        'reldi': {'errors': e_rel, 'diac_errors': d_rel,
                  'errors_on_undiacritized_gold': f_rel,
                  'share_pct': round(100 * f_rel / e_rel, 1)},
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\nSacuvano: {out}')


if __name__ == '__main__':
    main()
