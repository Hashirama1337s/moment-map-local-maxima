"""Driver: one group, one H.  Usage: python run_case.py <case> <seed> <nstarts> <nsaddle> [maxsample]
cases: u22, u222, ferm42, ferm52, bos32, bos42, spin1, spin32, spin2, spin52, spin_h1, coll2, coll3, u1_0112, su3adj
Output: results/<case>_<seed>.json and a log on stdout."""
import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_core import *

CASES = {
    'u22': lambda: g_product([2, 2]), 'u222': lambda: g_product([2, 2, 2]), 'u23': lambda: g_product([2, 3]),
    'ferm42': lambda: g_fermions(4, 2), 'ferm52': lambda: g_fermions(5, 2), 'ferm62': lambda: g_fermions(6, 2),
    'bos32': lambda: g_bosons(3, 2), 'bos42': lambda: g_bosons(4, 2), 'bos33': lambda: g_bosons(3, 3),
    'spin1': lambda: g_spin([1]), 'spin32': lambda: g_spin([1.5]), 'spin2': lambda: g_spin([2]), 'spin52': lambda: g_spin([2.5]),
    'spin_h1': lambda: g_spin([0.5, 1]),
    'coll2': lambda: g_collective(2), 'coll3': lambda: g_collective(3),
    'u1_0112': lambda: g_u1([0, 1, 1, 2]), 'su3adj': lambda: g_su3_adjoint(),
}

def main():
    case, seed, nstarts, nsaddle = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    maxsample = int(sys.argv[5]) if len(sys.argv) > 5 else 400
    t0 = time.time(); rng = np.random.default_rng(seed + 99)
    G = CASES[case](); d = G.d; H = gue(d, seed); lam = np.sort(np.linalg.eigvalsh(H)); nH2 = float(np.sum(lam ** 2))
    tol = 1e-8 * nH2; hb = herm_basis(d)
    kern_pred = d + G.nroot + int(np.sum(G.mult * (G.mult - 1)))
    R = dict(case=case, group=G.name, seed=seed, d=d, dimk=G.dimk, rank=G.rank, nroot=G.nroot, multfree=G.multfree,
             mult=G.mult.tolist(), W=G.W, kern_pred=kern_pred, spec=lam.tolist())
    print(f'[{case}] {G.name}: d={d} dim k={G.dimk} rank={G.rank} roots={G.nroot} multfree={G.multfree} mult={G.mult.tolist()} |W|={G.W}', flush=True)

    # ---- 1. enumerate T-fixed assignments and the predicted local-minimum classes
    n_all = 0; n_oi = 0; pred = {}; oi_perms = []; non_oi_sample = []
    for T, ellp, oi in assignments(G, lam):
        n_all += len(T); n_oi += int(oi.sum())
        for r in np.where(oi)[0]:
            tau = T[r]; w = G.Qt[tau]; key = make_key(ellp[r], w @ w.T)
            if key not in pred: pred[key] = tau.tolist()
        nz = np.where(~oi)[0]
        if len(nz): non_oi_sample += [T[i].tolist() for i in rng.choice(nz, size=min(len(nz), 50), replace=False)]
    R.update(n_assign=n_all, n_ordiso=n_oi, n_pred_classes=len(pred), oi_over_W=n_oi / G.W)
    print(f'  assignments {n_all}, order-iso {n_oi}, predicted classes {len(pred)} (order-iso/|W| = {n_oi / G.W:.3f})', flush=True)

    # ---- 2. Hessians at T-fixed points: order-iso => PSD with kernel kern_pred; otherwise a negative eigenvalue
    sample = list(pred.values()) + [non_oi_sample[i] for i in rng.permutation(len(non_oi_sample))[:maxsample]]
    hk = dict(n=0, oi_ok=0, oi_bad=0, non_ok=0, non_bad=0, block_err=0.0, min_neg_non=0.0, min_pos_oi=np.inf, bad=[])
    for tau in sample:
        E = D_from_perm(G, lam, tau); U = U_from_E(H, E); Hp, L = parts(G, U, H)
        assert np.abs(Hp - np.diag(E)).max() < 1e-10
        ev = np.linalg.eigvalsh(hess_analytic(G, Hp, L, hb)); evb = block_hessian_spectrum(G, E)
        hk['block_err'] = max(hk['block_err'], float(np.abs(ev - evb).max()))
        ell = G.Pdiag @ E; off = G.label[:, None] != G.label[None, :]
        oi = bool(np.min(((ell[:, None] - ell[None, :]) * (E[:, None] - E[None, :]))[off]) > 1e-9 * nH2) if off.any() else True
        nneg = int(np.sum(ev < -tol)); nzero = int(np.sum(np.abs(ev) <= tol)); hk['n'] += 1
        if oi:
            good = nneg == 0 and nzero == kern_pred; hk['oi_ok' if good else 'oi_bad'] += 1
            pos = ev[ev > tol]; hk['min_pos_oi'] = min(hk['min_pos_oi'], float(pos[0]) if len(pos) else np.inf)
        else:
            good = nneg > 0; hk['non_ok' if good else 'non_bad'] += 1
            hk['min_neg_non'] = min(hk['min_neg_non'], float(-ev[0]) if hk['min_neg_non'] == 0 else float(-ev[0]))
        if not good and len(hk['bad']) < 5: hk['bad'].append(dict(tau=list(map(int, tau)), oi=oi, nneg=nneg, nzero=nzero, ev0=float(ev[0])))
    hk['min_pos_oi'] = float(hk['min_pos_oi']); R['tfixed_hessian'] = hk
    print(f'  T-fixed Hessians: {hk["n"]} checked; order-iso ok {hk["oi_ok"]} bad {hk["oi_bad"]}; non-order-iso with neg eig {hk["non_ok"]} bad {hk["non_bad"]}; '
          f'block-formula err {hk["block_err"]:.2e}', flush=True)

    # ---- 3. autograd vs analytic Hessian (3 T-fixed points + 2 critical points from LM later) and Givens identity
    ag = 0.0
    for tau in sample[:3]:
        E = D_from_perm(G, lam, tau); U = U_from_E(H, E); Hp, L = parts(G, U, H)
        ag = max(ag, float(np.abs(hess_autograd(G, U, H, hb) - hess_analytic(G, Hp, L, hb)).max()))
    gv = 0.0; ngv = 0
    rootpair = np.zeros((d, d), bool)
    for r in G.roots:
        for (cp, c) in r['pairs']: rootpair[cp, c] = rootpair[c, cp] = True
    for tau in sample[:20]:
        E = D_from_perm(G, lam, tau); U = U_from_E(H, E); ell = G.Pdiag @ E; f0 = cost(G, U, H)
        cand = [(c, cp) for c in range(d) for cp in range(d) if c < cp and G.label[c] != G.label[cp] and not rootpair[c, cp]]
        if not cand: break
        c, cp = cand[rng.integers(len(cand))]; e = np.zeros(d); e[c], e[cp] = 1, -1
        pe = np.linalg.norm(G.Pdiag @ e) ** 2; Dl = E[c] - E[cp]
        for th in (0.05, 0.3, 1.0):
            Rg = np.eye(d, dtype=complex); Rg[c, c] = Rg[cp, cp] = np.cos(th); Rg[c, cp] = -np.sin(th); Rg[cp, c] = np.sin(th)
            pr = f0 + 2 * np.sin(th) ** 2 * Dl * (ell[c] - ell[cp]) - np.sin(th) ** 4 * Dl ** 2 * pe
            gv = max(gv, abs(cost(G, U @ Rg, H) - pr)); ngv += 1
    R.update(autograd_err=ag, givens_err=float(gv), givens_n=ngv)
    print(f'  autograd-vs-analytic {ag:.2e}; Givens identity err {gv:.2e} over {ngv} tests', flush=True)

    # ---- 4. batched descent from Haar starts
    found = []
    if nstarts:
        U0 = [haar(d, rng) for _ in range(nstarts)]
        Uf, ff, gn, its = descend_batch(G, H, U0)
        conv = gn < 1e-6 * nH2
        print(f'  descent: {nstarts} starts, {its} its, {int(conv.sum())} with grad<1e-6|H|^2, max grad {gn.max():.2e}', flush=True)
        stats = dict(n=nstarts, conv=int(conv.sum()), polished=0, minima=0, saddles=0, tfixed=0, ordiso=0, in_pred=0, kern_ok=0, viol=[])
        keys_found = {}
        for i in range(nstarts):
            U = Uf[i]
            if not conv[i]: continue
            U, nr = lm_critical(G, U, H, hb)
            c = classify(G, U, H, hb); stats['polished'] += 1
            if c['grad'] > 1e-9 * nH2: continue
            if not c['cand']: stats['saddles'] += 1; continue
            stats['minima'] += 1; stats['tfixed'] += c['tfixed']
            if c['tfixed']:
                stats['ordiso'] += c['ordiso']; stats['kern_ok'] += int(c['n_zero'] == kern_pred)
                match = [k for k in pred if np.allclose(k[0], c['key'][0], atol=2e-4) and np.allclose(k[1], c['key'][1], atol=2e-4)]
                stats['in_pred'] += bool(match)
                if match: keys_found[match[0]] = keys_found.get(match[0], 0) + 1
            if not (c['tfixed'] and c['ordiso']) and len(stats['viol']) < 5: stats['viol'].append({k: v for k, v in c.items() if k != 'key'})
        stats['classes_found'] = len(keys_found); stats['class_counts'] = sorted(keys_found.values(), reverse=True)
        fv = {}
        for k, tau in pred.items(): fv[k] = cost(G, U_from_E(H, D_from_perm(G, lam, tau)), H)
        stats['f_pred'] = sorted(fv.values()); stats['f_found'] = sorted(fv[k] for k in keys_found)
        R['descent'] = stats
        print(f'  minima {stats["minima"]} (T-fixed {stats["tfixed"]}, order-iso {stats["ordiso"]}, kernel ok {stats["kern_ok"]}, in predicted set {stats["in_pred"]}); '
              f'classes found {stats["classes_found"]}/{len(pred)}; saddles {stats["saddles"]}', flush=True)

    # ---- 5. LM critical-point search (any index): Lemma A (N1 >= 0 => T-fixed) and PSD => T-fixed & order-iso
    if nsaddle:
        sd = dict(n=nsaddle, conv=0, tfixed=0, non_tfixed=0, N1_nonneg=0, N1_nonneg_not_tfixed=0, cand=0, cand_bad=0,
                  min_N1_non_tfixed=0.0, max_hmin_non_tfixed=-np.inf, ag_err=0.0, examples=[])
        for i in range(nsaddle):
            U, nr = lm_critical(G, haar(d, rng), H, hb)
            if nr > 1e-11 * nH2: continue
            c = classify(G, U, H, hb); sd['conv'] += 1
            sd['tfixed'] += c['tfixed']; sd['non_tfixed'] += (not c['tfixed'])
            if c['N1'] >= -tol:
                sd['N1_nonneg'] += 1; sd['N1_nonneg_not_tfixed'] += (not c['tfixed'])
            if not c['tfixed']:
                sd['min_N1_non_tfixed'] = min(sd['min_N1_non_tfixed'], c['N1']) if sd['min_N1_non_tfixed'] else c['N1']
                sd['max_hmin_non_tfixed'] = max(sd['max_hmin_non_tfixed'], c['hess_min'])
                if c['N1'] > sd.get('_n1best', -np.inf): sd['_n1best'] = c['N1']
            if c['cand']:
                sd['cand'] += 1
                if not (c['tfixed'] and c['ordiso']): sd['cand_bad'] += 1
            if i < 2:
                Hp, L = parts(G, U, H); sd['ag_err'] = max(sd['ag_err'], float(np.abs(hess_autograd(G, U, H, hb) - hess_analytic(G, Hp, L, hb)).max()))
        sd['max_N1_non_tfixed'] = sd.pop('_n1best', None)
        R['lm'] = sd
        print(f'  LM: {sd["conv"]}/{nsaddle} converged; T-fixed {sd["tfixed"]}, non-T-fixed {sd["non_tfixed"]} '
              f'(max N1 among non-T-fixed {sd["max_N1_non_tfixed"]}, max Hess-min {sd["max_hmin_non_tfixed"]:.3e}); '
              f'N1>=0 & not T-fixed: {sd["N1_nonneg_not_tfixed"]}; PSD points {sd["cand"]}, violating {sd["cand_bad"]}; autograd err {sd["ag_err"]:.1e}', flush=True)
    R['seconds'] = time.time() - t0
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', f'{case}_{seed}.json'), 'w') as fh:
        json.dump(R, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
    print(f'  done in {R["seconds"]:.0f}s', flush=True)

if __name__ == '__main__':
    main()
