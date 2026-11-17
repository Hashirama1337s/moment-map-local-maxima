"""Figures for the moment-map / stable-splits repository, drawn only from files the repository ships.

usage:  python figures/make_figures.py [repo_root]      (repo_root defaults to the folder above this script)
writes: figures/grids_3x3.svg, figures/worked_example.svg, figures/bracket_4x4.svg  (next to this script)

Sources: theory/results/farkas3x3.json (all 42 3x3 grids, witnesses and certificates), theory/results/mi_census.json,
theory/results/d5lp/certificates/{lower_1209,upper_cores}.json. Every witness and certificate drawn is re-checked here in
exact integer arithmetic; the script stops if one fails.
Checked copy (figure referee, 2026-11-17): the 2,845 label on the 4x4 bracket is right-aligned so it no longer runs into
the 3,330 label.
"""
import itertools, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.dirname(HERE)
OUT = HERE if os.path.basename(HERE) == "figures" else os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---- lab style (STYLE.md) -------------------------------------------------------------------------------------------
BG, BORDER, GRID, AX, TXT, ACC = "#1c2026", "#2f353d", "#22272e", "#9aa4ae", "#e8edf2", "#39ff14"
CAT = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"]
NEUTRAL = "#5b636c"
plt.rcParams.update({
    "svg.fonttype": "none", "svg.hashsalt": "moment-map-local-maxima",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 14, "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": AX, "axes.labelcolor": TXT, "text.color": TXT, "xtick.color": AX, "ytick.color": AX,
    "axes.titlecolor": TXT, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
})
W_IN = 12.5


def frame(fig):
    fig.patches.append(FancyBboxPatch((0.004, 0.004), 0.992, 0.992, boxstyle="round,pad=0,rounding_size=0.012",
                                      transform=fig.transFigure, fill=False, ec=BORDER, lw=1.5, zorder=-1))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), format="svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)
    print("wrote figures/" + name)


def J(*p):
    return json.load(open(os.path.join(ROOT, *p)))


# ---- exact checks (integers only) -----------------------------------------------------------------------------------------
def cells_by_rank(T):
    return sorted(((T[a][b], (a, b)) for a in range(len(T)) for b in range(len(T[0]))))


def is_syt(T):
    m, n = len(T), len(T[0])
    return (sorted(v for r in T for v in r) == list(range(m * n))
            and all(T[a][b] < T[a][b + 1] for a in range(m) for b in range(n - 1))
            and all(T[a][b] < T[a + 1][b] for a in range(m - 1) for b in range(n)))


def witness_ok(T, x, y):
    vals = [x[a] + y[b] for _, (a, b) in cells_by_rank(T)]
    return all(u < v for u, v in zip(vals, vals[1:]))


def certificate_ok(T, cert):
    """each item says L(smaller) < L(larger) is forced by the order of T; the weighted sum must be the zero form."""
    m, n = len(T), len(T[0])
    tot = [0] * (m + n)
    for c in cert:
        (a1, b1), (a2, b2), w = c["smaller"], c["larger"], c["multiplier"]
        assert T[a1][b1] < T[a2][b2] and w > 0
        tot[a2] += w; tot[m + b2] += w; tot[a1] -= w; tot[m + b1] -= w
    return all(t == 0 for t in tot)


def all_syt_3x3():
    out = []
    for perm in itertools.permutations(range(9)):
        T = [list(perm[0:3]), list(perm[3:6]), list(perm[6:9])]
        if is_syt(T):
            out.append(T)
    return out


def transpose(T):
    return [list(r) for r in zip(*T)]


# ---- figure 1: the 42 grids -----------------------------------------------------------------------------------------------
def draw_grid(ax, x0, y0, T, size, edge, fs, fill=None):
    m = len(T)
    for a in range(m):
        for b in range(len(T[0])):
            ax.add_patch(Rectangle((x0 + b * size, y0 - (a + 1) * size), size, size, fc=fill or BG, ec=edge, lw=1.3))
            ax.text(x0 + (b + 0.5) * size, y0 - (a + 0.5) * size, str(T[a][b] + 1), ha="center", va="center",
                    fontsize=fs, color=TXT)


def fig_grids():
    d = J("theory", "results", "farkas3x3.json")["3x3"]
    real, nonreal = d["realizable"], d["nonrealizable"]
    for r in real:
        assert is_syt(r["T"]) and witness_ok(r["T"], r["x"], r["y"])
    for r in nonreal:
        assert is_syt(r["T"]) and certificate_ok(r["T"], r["pair_certificate"])
    allT = all_syt_3x3()
    shipped = sorted([r["T"] for r in real] + [r["T"] for r in nonreal])
    assert sorted(allT) == shipped and len(allT) == d["n_syt"] == 42      # the file lists every 3x3 grid exactly once
    keys = {json.dumps(r["T"]) for r in real}
    classes = {min(json.dumps(r["T"]), json.dumps(transpose(r["T"]))) for r in real}
    assert all(json.dumps(transpose(r["T"])) in keys for r in real)
    n_classes = len(classes)
    census = J("theory", "results", "mi_census.json")["3x3"]["classes"]
    assert n_classes == d["n_transpose_classes_realizable"] == census == 18

    fig = plt.figure(figsize=(W_IN, 9.8))
    frame(fig)
    fig.text(0.03, 0.945, f"All {len(allT)} ways to rank a 3 x 3 grid: {len(real)} can be stable splits, {len(nonreal)} never can",
             fontsize=20, weight="bold")
    fig.text(0.03, 0.905, "Energies ranked 1 (lowest) to 9 and seated in a grid so that every row and column increases. "
                         "A ranking can be stable only if", fontsize=13, color=AX)
    fig.text(0.03, 0.876, "some \"row number + column number\" puts the cells in the same order; for 6 grids no numbers "
                          "can, and a short certificate proves it.", fontsize=13, color=AX)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.80]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * 0.80 * 9.8 / (0.96 * W_IN))
    ax.set_aspect("equal"); ax.axis("off")
    s, gap = 2.65, 2.05                     # cell size and gap, in axis units (square cells)
    blk = 3 * s + gap
    for i, r in enumerate(real):            # 6 x 6 block
        row, col = divmod(i, 6)
        draw_grid(ax, 2 + col * blk, 61 - row * blk, r["T"], s, CAT[0], 11)
    ax.text(2 + 3 * blk - gap / 2, 62.6, f"{len(real)} possible: some row + column numbers fit", ha="center", fontsize=14,
            color=CAT[0], weight="bold")
    xo = 2 + 6 * blk + 5
    for i, r in enumerate(nonreal):         # 3 x 2 block
        row, col = divmod(i, 2)
        draw_grid(ax, xo + col * blk, 61 - row * blk, r["T"], s, CAT[1], 11)
    ax.text(xo + blk - gap / 2, 62.6, f"{len(nonreal)} impossible", ha="center", fontsize=14, color=CAT[1], weight="bold")
    ax.text(xo, 61 - 3 * blk - 1.5,
            f"A grid and its mirror image\n(swap rows and columns) are\nthe same split, so the {len(real)}\n"
            f"possible grids give {n_classes} kinds\nof stable split: the count\nfound earlier by computer.",
            ha="left", va="top", fontsize=13.5, color=TXT, linespacing=1.35)
    save(fig, "grids_3x3.svg")
    return len(allT), len(real), len(nonreal), n_classes


# ---- figure 2: the worked example ------------------------------------------------------------------------------------------
def fig_worked():
    d = J("theory", "results", "farkas3x3.json")["3x3"]
    good = next(r for r in d["realizable"] if r["T"] == [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    bad = next(r for r in d["nonrealizable"] if r.get("T_1based") == "126/347/589")
    assert witness_ok(good["T"], good["x"], good["y"]) and certificate_ok(bad["T"], bad["pair_certificate"])
    x, y = good["x"], good["y"]

    fig = plt.figure(figsize=(W_IN, 6.6))
    frame(fig)
    fig.text(0.03, 0.925, "How the test works: one grid that can be stable, one that never can", fontsize=20, weight="bold")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84]); ax.set_xlim(0, 100); ax.set_ylim(0, 100 * 0.84 * 6.6 / (0.96 * W_IN))
    ax.set_aspect("equal"); ax.axis("off")
    s = 9.0
    # left: the witness
    gx, gy = 14, 41
    draw_grid(ax, gx, gy, good["T"], s, CAT[0], 20)
    for a in range(3):
        ax.text(gx - 2.2, gy - (a + 0.5) * s, f"x = {x[a]}", ha="right", va="center", fontsize=14, color=CAT[0])
    for b in range(3):
        ax.text(gx + (b + 0.5) * s, gy + 2.2, f"y = {y[b]}", ha="center", va="bottom", fontsize=14, color=CAT[0])
    for a in range(3):
        for b in range(3):
            ax.text(gx + (b + 0.9) * s, gy - (a + 0.86) * s, f"{x[a] + y[b]}", ha="right", va="center", fontsize=11,
                    color=AX)
    ax.text(gx + 1.5 * s, gy - 3 * s - 3.0, "123 / 456 / 789: possible", ha="center", va="top", fontsize=15, weight="bold",
            color=CAT[0])
    ax.text(gx + 1.5 * s, gy - 3 * s - 7.0, "Row number x plus column number y\n(small grey numbers) rises in the same "
            "order\nas the ranks 1 to 9.", ha="center", va="top", fontsize=13, color=TXT, linespacing=1.3)
    # right: the certificate
    bx, by = 64, 41
    draw_grid(ax, bx, by, bad["T"], s, CAT[1], 20)

    def centre(c):
        a, b = c
        return (bx + (b + 0.5) * s, by - (a + 0.5) * s)

    for c in bad["pair_certificate"]:
        p, q = centre(c["smaller"]), centre(c["larger"])
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=18, lw=2, color=ACC, shrinkA=16, shrinkB=16,
                                     connectionstyle="arc3,rad=0.18", zorder=5))
    ax.text(bx + 1.5 * s, by - 3 * s - 3.0, "126 / 347 / 589: impossible", ha="center", va="top", fontsize=15, weight="bold",
            color=CAT[1])
    ax.text(bx + 1.5 * s, by - 3 * s - 7.0, "Each arrow runs from a lower rank to a higher one, so\n\"row + column\" must be smaller where it starts. "
            "Added\nup, the three demands say \"something < itself\".", ha="center", va="top",
            fontsize=13, color=TXT, linespacing=1.3)
    save(fig, "worked_example.svg")
    return x, y, bad["pair_certificate"]


# ---- figure 3: the 4 x 4 bracket ---------------------------------------------------------------------------------------------
def fig_bracket():
    lo = J("theory", "results", "d5lp", "certificates", "lower_1209.json")
    up = J("theory", "results", "d5lp", "certificates", "upper_cores.json")
    assert lo["shape"] == up["shape"] == "4x4"
    L, U, T, nc = lo["claimed_realms"], up["upper_bound"], up["total_realms"], up["n_cores"]
    assert U == T - nc and len(up["cores"]) == nc

    fig, ax = plt.subplots(figsize=(W_IN, 5.4))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.70, bottom=0.20)
    frame(fig)
    fig.text(0.03, 0.90, "Split 16 levels into two 4-level parts: how many stable splits can one system have?",
             fontsize=19, weight="bold")
    fig.text(0.03, 0.835, f"There are {T:,} kinds of stable split in all, but no single list of 16 energies makes them all "
                          "stable at once.", fontsize=13.5, color=AX)
    fig.text(0.03, 0.785, "The largest number one list can achieve is proved to lie in the shaded range; its exact value "
                          "is open.", fontsize=13.5, color=AX)
    ax.axvspan(L, U, ymin=0.25, ymax=0.75, color=CAT[0], alpha=0.30, lw=0)
    ax.plot([L, U], [0.5, 0.5], color=CAT[0], lw=2)
    for v in (L, U):
        ax.plot([v, v], [0.25, 0.75], color=CAT[0], lw=2)
    ax.plot([T, T], [0.25, 0.75], color=TXT, lw=2)
    ax.text(L, 0.82, f"{L:,}\nachieved by an explicit\nlist of energies (exact check)", ha="center", va="bottom", fontsize=12.5,
            color=TXT, linespacing=1.25)
    ax.text(U + 40, 0.82, f"{U:,} = {T:,} - {nc}\n{nc} separate groups, each\nmust lose at least one", ha="right", va="bottom",
            fontsize=12.5, color=TXT, linespacing=1.25)
    ax.text(T, 0.82, f"{T:,}\nall kinds of\nstable split", ha="center", va="bottom", fontsize=12.5, color=TXT, linespacing=1.25)
    ax.text((L + U) / 2, 0.36, "true maximum lies here", ha="center", va="center", fontsize=13.5, color=TXT)
    ax.set_xlim(0, 3500); ax.set_ylim(0, 1.0)
    ax.set_yticks([])
    for sname in ("top", "right", "left"):
        ax.spines[sname].set_visible(False)
    ax.set_xticks(range(0, 3501, 500))
    ax.set_xticklabels([f"{t:,}" for t in range(0, 3501, 500)])
    ax.set_xlabel("number of stable splits at one list of 16 energies")
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.set_clip_on(False)
    save(fig, "bracket_4x4.svg")
    return L, U, T, nc


if __name__ == "__main__":
    print("  grids:", fig_grids())
    print("  worked example:", fig_worked())
    print("  bracket:", fig_bracket())
