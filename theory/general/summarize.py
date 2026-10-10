"""Aggregate results/*.json (run_case outputs) into results/summary.txt."""
import json, glob, os
D = os.path.dirname(os.path.abspath(__file__)); rows = []
hdr = ('case seed d dimk rank mult |W| assign ordiso Korbits keycls | Hess: oi_ok/oi_bad non_neg/non_bad blockerr | '
       'descent: starts minima Tfixed ordiso kern_ok inpred classes | LM: conv nonT N1>=0&nonT PSD PSDbad')
rows.append(hdr)
for f in sorted(glob.glob(os.path.join(D, 'results', '*_1[0-9].json'))):
    R = json.load(open(f))
    if 'tfixed_hessian' not in R: continue
    h = R['tfixed_hessian']; s = R.get('descent', {}); l = R.get('lm', {})
    rows.append(f"{R['case']} {R['seed']} {R['d']} {R['dimk']} {R['rank']} {''.join(map(str, R['mult'])) if not R['multfree'] else 'mf'} {R['W']} "
                f"{R['n_assign']} {R['n_ordiso']} {R['oi_over_W']:.0f} {R['n_pred_classes']} | {h['oi_ok']}/{h['oi_bad']} {h['non_ok']}/{h['non_bad']} {h['block_err']:.1e} | "
                f"{s.get('n')} {s.get('minima')} {s.get('tfixed')} {s.get('ordiso')} {s.get('kern_ok')} {s.get('in_pred')} {s.get('classes_found')}/{R['n_pred_classes']} | "
                f"{l.get('conv')} {l.get('non_tfixed')} {l.get('N1_nonneg_not_tfixed')} {l.get('cand')} {l.get('cand_bad')}")
open(os.path.join(D, 'results', 'summary.txt'), 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))
