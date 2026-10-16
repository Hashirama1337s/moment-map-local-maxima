# BAR-D13C Amendment 1 (sealed before the scored run; BAR-D13C.md itself unchanged)

**In plain words.** The sealed tie rule (relative gap <= 1e-12 = "tie") was wrong for this regime: at beta = 200 the exact
winner and runner-up can differ only through high levels, so their MI differs by a relative 1e-20 or less while both are
computed to 100+ correct digits. A 5+1-instance probe (seeds 0-4 of 2x4 GUE, seed 0 of 4x4 GUE) flagged the 4x4 instance as a
"tie" under the old rule. I did not read the probe's P1/P2 values; this amendment changes only the tie definition.

Change: an instance is a TIE (excluded from P1/P2, counted) iff the dps + 60 recomputation of winner and runner-up does not
reproduce the order, or the relative gap is below 10^-(dps - 20), where dps is the working precision at beta = 200.
Everything else in BAR-D13C (instances, beta, bars, P1-P3, descriptive items) is unchanged.
