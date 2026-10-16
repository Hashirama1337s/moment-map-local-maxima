# BAR-D13C (sealed before the run). Door 13 cold-limit law: which split wins when the system is cold?

**In plain words.** Door 13 found that the least-correlated split of a thermal state changes as it cools, but the cold end was
never resolved (float underflow). The recon seat derived a leading-order law: near zero temperature the mutual information of an
arrangement decays like exp(-beta X) with X = min(e11, e10 + e01) (excitation energies above the ground cell (0,0)), so the cold
winner should be an arrangement that maximises X: the lowest excitations each live in one part, and the first "mixed" level
(1,1) comes as late as possible. We test it against exact high-precision optima.

## Object (fixed)
- Same instances and arrangement sets as d13_thermal.py: spectra normalised to zero mean, unit std; all SYT of the grid,
  transposes identified for square grids (same dedupe rule). p_ab ~ exp(-beta (E_ab - E_min)). MI = H(A) + H(B) - H(AB).
- Exact MI: mpmath (gmpy backend), dps = 40 + ceil(beta (E_max - E_min) / ln 10). Winner = argmin exact MI over all classes.
- X(class) = min(e_(1,1), e_(1,0) + e_(0,1)), e = E - E_min at those cells. Leading formula (seat B):
  MI_lead = sum_{a,b>=1} [p_ab ln(p_ab / (p_a0 p_0b)) - p_ab + p_a0 p_0b] with normalised p.
- Instances: 2x4 GUE seeds 70000-70199 (200); 3x3 GUE 70000-70199 (200); 4x4 GUE 70000-70049 (50); 4x4 local mixed-field
  Ising seeds 80000-80049 (50). Generators d8_exact.gue_spectrum, bmargin.ising_spectrum (as door 13). Scored beta = 200.

## Controls (VOID, not FAILED, if one fails)
- C1 (instrument): at beta = 1, mpmath MI equals the door-13 float MI to relative 1e-9 for every class, first 5 instances per set.
- C2 (precision): at beta = 200 the winner and runner-up are recomputed at dps + 60; same winner, and the relative gap
  (MI_2 - MI_1) / MI_1 exceeds 1e-12 (else the instance is reported as an exact tie, excluded from P1/P2, counted).
- C3 (inputs): the 4x4 GUE / Ising instances equal results/d13/blind_inputs_4x4_{GUE,ISING}.npy (first 50) to 1e-12.

## Predictions (scored per set: 2x4 GUE, 3x3 GUE, 4x4 GUE, 4x4 Ising)
- P1: the exact beta = 200 winner lies in the argmax-X set in >= 95% of non-tied instances. FAILED if < 95% in any set.
- P2: the exact beta = 200 winner equals argmin MI_lead(beta = 200) in >= 95% of non-tied instances. FAILED if < 95% in any set.
- P3: at the exact winner, |MI_exact / MI_lead - 1| < 0.1 in >= 95% of instances. FAILED if < 95% in any set.

## Descriptive (not scored)
- Chance baseline: mean |argmax-X set| / classes. Share of instances where e11 binds (e11 < e10 + e01).
- Winner stability across beta = 50, 100, 200. Agreement with the door-13 float cold winner at beta = 100.
Code: theory/d13_cold.py. Interpreter py -3.11.
