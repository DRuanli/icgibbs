"""
make_tables.py — every number and table in the paper, generated from the analysis outputs (never typed by hand).

Inputs (all produced by the registered analysis scripts from raw results):
  G8/results/g8_confirmatory.json, g8_descriptive.json, table_*.csv     Study 1 (TabPFN v2, confirmatory) + exploratory
  S2/results/s2_confirmatory.json, s2_table_*.csv                        Study 2 (TabICL v2, confirmatory, new datasets)
Outputs:
  paper/numbers.tex                 \\val{key} macros used in the text
  paper/tables/main_results.tex     Table: pre-registered tests, Study 1 vs Study 2
  paper/tables/h3.tex               Table: Gibbs vs hard-EM (same TFM, same warm start)
  paper/tables/s2_means.tex         Appendix table: mean score of every method, Study 2
  paper/tables/r4.tex               Appendix table: Study 2 tests with a second chain and with two chains averaged
                                    (exploratory lot R4; V2/results/v0_summary.json from the pre-written v0_report.py,
                                    V2/results/v0_r4.json from v0_explore.py E4)
Run:  python paper/scripts/make_tables.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
ROOT = os.path.dirname(PAPER)
G8 = os.path.join(ROOT, "G8", "results")
S2 = os.path.join(ROOT, "S2", "results")

NAMES = {
    "gibbs_warm": r"\textbf{In-Context Gibbs (ours)}", "hard_em_warm": "Hard-EM (same TFM)",
    "sem_warm": "Stochastic EM (same TFM)", "naive": "Naive TFM", "oracle": "Oracle (true labels)",
    "dllp": "DLLP (MLP)", "dllplin": "DLLP (linear)", "llpvat": "LLP-VAT", "easyllp": "EasyLLP", "glws": "GLWS",
    "proden": "PRODEN", "glwspll": "GLWS", "rc": "RC", "cc": "CC",
}


def f3(x, sign=True):
    s = f"{x:+.3f}" if sign else f"{x:.3f}"
    return s.replace("-", r"\ensuremath{-}")


def fp(p):
    if p < 0.001:
        return r"\ensuremath{<}0.001"
    return f"{p:.3f}"


def prel(p):
    """p with its relation sign, for use after p_Holm in running text"""
    return r"\ensuremath{<}0.001" if p < 0.001 else r"\ensuremath{=}" + f"{p:.3f}"


def wtl(c):
    return f"{c['wins']}/{c['ties']}/{c['losses']}"


def ci(c):
    return f"[{f3(c['ci_lo'])}, {f3(c['ci_hi'])}]"


def load():
    s1 = json.load(open(os.path.join(G8, "g8_confirmatory.json")))
    s2 = json.load(open(os.path.join(S2, "s2_confirmatory.json")))
    return s1, s2


def numbers(s1, s2):
    v = {}

    def put(prefix, c):
        v[prefix + "-diff"] = f3(c["mean_diff"])
        v[prefix + "-ci"] = ci(c)
        v[prefix + "-wtl"] = wtl(c)
        v[prefix + "-w"] = str(c["wins"])
        v[prefix + "-n"] = str(c["n"])
        if "p_holm" in c:
            v[prefix + "-p"] = fp(c["p_holm"])
            v[prefix + "-prel"] = prel(c["p_holm"])

    for tag, js in (("s1", s1), ("s2", s2)):
        put(f"{tag}-llp-cmp", js["H2_llp8"]["comparator"])
        put(f"{tag}-llp-naive", js["H2_llp8"]["naive"])
        for q in ("0.3", "0.6"):
            put(f"{tag}-pll{q[-1]}-cmp", js["H4_pll"][q]["comparator"])
            put(f"{tag}-pll{q[-1]}-naive", js["H4_pll"][q]["naive"])
            for k, m in js["H4_pll"][q]["means"].items():
                v[f"{tag}-pll{q[-1]}-mean-{k}"] = f"{m:.3f}"
        for k, m in js["H2_llp8"]["means"].items():
            v[f"{tag}-llp-mean-{k}"] = f"{m:.3f}"
    for q in ("0.3", "0.6"):                                 # gap to oracle, accuracy points
        m = s2["H4_pll"][q]["means"]
        v[f"s2-pll{q[-1]}-oracle-gap"] = f"{100 * (m['oracle'] - m['gibbs_warm']):.1f}"
    v["s2-llp-oracle-gap"] = f"{s2['H2_llp8']['means']['oracle'] - s2['H2_llp8']['means']['gibbs_warm']:.3f}"
    v["s2-llp-strongest-name"] = {"dllp_tun": "tuned DLLP (MLP)", "dllplin_tun": "tuned DLLP (linear)",
                                  "llpvat_tun": "tuned LLP-VAT"}.get(s2["H2_llp8"]["strongest"]["name"], s2["H2_llp8"]["strongest"]["name"])
    st = s2["H2_llp8"]["strongest"]
    v["s2-llp-strongest-wins"] = f"{st['wins']}/{st['n']}"
    v["s2-llp-strongest-frac"] = f"{100 * s2['H2_llp8']['frac_wins_strongest']:.0f}\\%"
    for q in ("0.3", "0.6"):
        st = s2["H4_pll"][q]["strongest"]
        v[f"s2-pll{q[-1]}-rule60"] = f"{st['n_ge']}/{st['n_datasets_level']}"
    h = s2["H3"]
    v["s2-h3-cells"] = str(h["n_cells"])
    v["s2-h3-frac"] = f"{100 * h['frac_gibbs_better']:.1f}\\%"
    v["s2-h3-rho"] = f"{h['spearman_rho_pll']:.2f}"
    v["s2-h3-prope"] = f"{h['bayes']['p_rope']:.2f}"
    h = s1["H3"]
    v["s1-h3-cells"] = str(h["n_cells"])
    v["s1-h3-frac"] = f"{100 * h['frac_gibbs_better']:.1f}\\%"
    v["s1-h3-rho"] = f"{h['spearman_rho']:.2f}"
    v["s1-n-binary"] = str(s1["H2_llp8"]["n_datasets"])
    v["s1-n-multi"] = str(s1["H4_pll"]["n_datasets"]) if "n_datasets" in s1["H4_pll"] else "14"
    v["s2-n-binary"] = str(s2["H2_llp8"]["n_datasets"])
    v["s2-n-multi"] = str(s2["H4_pll"]["0.3"]["n_datasets"])
    v["s2-n-total"] = str(int(v["s2-n-binary"]) + int(v["s2-n-multi"]))
    return v


def write_numbers(v):
    lines = ["% generated by paper/scripts/make_tables.py — do not edit", r"\makeatletter"]
    lines += [f"\\@namedef{{v@{k}}}{{{x}}}" for k, x in sorted(v.items())]
    lines += [r"\newcommand{\val}[1]{\@ifundefined{v@#1}{\textbf{\textcolor{red}{??#1}}}{\@nameuse{v@#1}}}", r"\makeatother"]
    open(os.path.join(PAPER, "numbers.tex"), "w").write("\n".join(lines) + "\n")


def main_table(s1, s2):
    rows = [("LLP, bag size 8", "AUC", "DLLP (linear)", ("H2_llp8", None, "comparator")),
            ("", "", "Naive TFM", ("H2_llp8", None, "naive")),
            ("PLL, $q=0.3$", "Acc.", "CC (tuned)", ("H4_pll", "0.3", "comparator")),
            ("", "", "Naive TFM", ("H4_pll", "0.3", "naive")),
            ("PLL, $q=0.6$", "Acc.", "CC (tuned)", ("H4_pll", "0.6", "comparator")),
            ("", "", "Naive TFM", ("H4_pll", "0.6", "naive"))]

    def get(js, key):
        h, q, c = key
        return js[h][c] if q is None else js[h][q][c]

    out = [r"% generated by paper/scripts/make_tables.py — do not edit",
           r"\begin{tabular}{@{}lllrrrrrr@{}}", r"\toprule",
           r" & & & \multicolumn{3}{c}{Study 1: TabPFN v2} & \multicolumn{3}{c}{Study 2: TabICL v2, new datasets} \\",
           r"\cmidrule(lr){4-6}\cmidrule(l){7-9}",
           r"Setting & Metric & Gibbs vs. & $\Delta$ & W/T/L & $p_{\mathrm{Holm}}$ & $\Delta$ [95\% CI] & W/T/L & $p_{\mathrm{Holm}}$ \\",
           r"\midrule"]
    for i, (st, met, comp, key) in enumerate(rows):
        a, b = get(s1, key), get(s2, key)
        sig = lambda c: (r"\textbf{" + fp(c["p_holm"]) + "}") if c["p_holm"] < 0.05 and c["mean_diff"] > 0 else fp(c["p_holm"])
        out.append(f"{st} & {met} & {comp} & {f3(a['mean_diff'])} & {wtl(a)} & {sig(a)} & "
                   f"{f3(b['mean_diff'])} {ci(b)} & {wtl(b)} & {sig(b)} \\\\")
        if i in (1, 3):
            out.append(r"\addlinespace")
    out += [r"\midrule",
            f"\\multicolumn{{3}}{{@{{}}l}}{{Pre-registered verdict}} & \\multicolumn{{3}}{{l}}{{H2: {'met' if s1['H2_llp8']['pass'] else 'not met'}; "
            f"H4: {'met' if s1['H4_pll'].get('weakest_reading_passes') else 'not met'}}} & "
            f"\\multicolumn{{3}}{{l}}{{H2$'$: {'met' if s2['H2_llp8']['pass'] else 'not met'}; H4$'$: {'met' if s2['H4_pll']['pass'] else 'not met'}}} \\\\",
            r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "main_results.tex"), "w").write("\n".join(out) + "\n")


def h3_table(s1, s2):
    a, b = s1["H3"], s2["H3"]
    out = [r"% generated by paper/scripts/make_tables.py — do not edit",
           r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
           r"Study (TFM; settings) & cells & Gibbs better & $P(\text{Gibbs better})$ & $P(\text{ROPE})$ & Spearman $\rho$ \\",
           r"\midrule",
           f"Study 1 (TabPFN v2; all six settings) & {a['n_cells']} & {100 * a['frac_gibbs_better']:.1f}\\% & "
           f"{a['bayes']['p_right']:.2f} & {a['bayes']['p_rope']:.2f} & {a['spearman_rho']:.2f} \\\\",
           f"Study 2 (TabICL v2; LLP bag 8, PLL) & {b['n_cells']} & {100 * b['frac_gibbs_better']:.1f}\\% & "
           f"{b['bayes']['p_right']:.2f} & {b['bayes']['p_rope']:.2f} & {b['spearman_rho_pll']:.2f}$^\\dagger$ \\\\",
           r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "h3.tex"), "w").write("\n".join(out) + "\n")


def means_table():
    l = pd.read_csv(os.path.join(S2, "s2_table_llp8.csv"), index_col=0)
    p = {q: pd.read_csv(os.path.join(S2, f"s2_table_pll_{q}.csv"), index_col=0) for q in (0.3, 0.6)}
    tfm = ["gibbs_warm", "hard_em_warm", "sem_warm", "naive"]
    rows = [r"% generated by paper/scripts/make_tables.py — do not edit",
            r"\begin{tabular}{@{}lrr@{\hspace{2.2em}}lrrrr@{}}", r"\toprule",
            r"\multicolumn{3}{@{}l}{LLP, bag 8 (AUC, 18 datasets)} & \multicolumn{5}{l}{PLL (accuracy, 10 datasets)} \\",
            r"\cmidrule(r){1-3}\cmidrule(l){4-8}",
            r"Method & default & tuned & Method & \multicolumn{2}{c}{$q=0.3$} & \multicolumn{2}{c}{$q=0.6$} \\",
            r" & & & & default & tuned & default & tuned \\", r"\midrule"]
    left = [(NAMES[m], f"{l[m].mean():.3f}", "") for m in tfm] + \
           [(NAMES[b], f"{l[b + '_def'].mean():.3f}", f"{l[b + '_tun'].mean():.3f}")
            for b in ("dllp", "dllplin", "llpvat", "easyllp", "glws")] + [(NAMES["oracle"], f"{l['oracle'].mean():.3f}", "")]
    right = [(NAMES[m], f"{p[0.3][m].mean():.3f}", "", f"{p[0.6][m].mean():.3f}", "") for m in tfm] + \
            [(NAMES[b], f"{p[0.3][b + '_def'].mean():.3f}", f"{p[0.3][b + '_tun'].mean():.3f}",
              f"{p[0.6][b + '_def'].mean():.3f}", f"{p[0.6][b + '_tun'].mean():.3f}") for b in ("proden", "glwspll", "rc", "cc")] + \
            [(NAMES["oracle"], f"{p[0.3]['oracle'].mean():.3f}", "", f"{p[0.6]['oracle'].mean():.3f}", "")]
    n = max(len(left), len(right))
    left += [("", "", "")] * (n - len(left))
    right += [("", "", "", "", "")] * (n - len(right))
    for (a, b, c), (d, e, f, g, h) in zip(left, right):
        rows.append(f"{a} & {b} & {c} & {d} & {e} & {f} & {g} & {h} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "s2_means.tex"), "w").write("\n".join(rows) + "\n")


def extra_numbers(v):
    """ablations, R-hat, real PLL and cost (Study 1: G8 outputs and G7 raw; Study 2: S2 raw)."""
    import glob
    import sys
    sys.path.insert(0, os.path.join(ROOT, "G8", "analysis"))
    from g8_confirmatory import load as load_s1
    ab = pd.read_csv(os.path.join(G8, "ablation_summary.csv"))
    ab["level"] = ab.level.astype(float)

    def row(var, st, lv):
        r = ab[(ab.variant == var) & (ab.setting == st) & (ab.level == lv)].iloc[0]
        return r

    for var, tag in (("gibbs_warm_mh", "mh"), ("gibbs_warm_K5", "k5"), ("gibbs_warm_K20", "k20"),
                     ("gibbs_warm_S15", "s15"), ("gibbs_warm_S60", "s60")):
        for st, lv, t2 in (("llp", 8.0, "llp"), ("pll", 0.3, "pll3")):
            r = row(var, st, lv)
            v[f"s1-abl-{tag}-{t2}"] = f3(r.mean_diff)
            v[f"s1-abl-{tag}-{t2}-wtl"] = f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
    r = row("gibbs_warm_trans", "llp", 8.0)
    v["s1-abl-trans-llp"] = f3(r.mean_diff)
    v["s1-abl-trans-llp-wtl"] = f"{int(r.wins)}/{int(r.ties)}/{int(r.losses)}"
    v["s1-abl-trans-llp-ll"] = f"{-r.mean_diff_logloss:.3f}"           # log loss worse by
    for st, lv, t2 in (("llp", 8.0, "llp"), ("pll", 0.3, "pll3"), ("pll", 0.6, "pll6")):
        r = row("gibbs_cold", st, lv)
        v[f"s1-abl-cold-{t2}"] = f3(r.mean_diff)
    rh = pd.read_csv(os.path.join(G8, "rhat_summary.csv"))
    rh["level"] = rh.level.astype(float)
    for st, lv, t2 in (("llp", 8.0, "llp"), ("pll", 0.3, "pll3"), ("pll", 0.6, "pll6")):
        r = rh[(rh.setting == st) & (rh.level == lv)].iloc[0]
        v[f"s1-rhat-{t2}-frac"] = f"{100 * r.frac_gt_1_1_median:.0f}\\%"
        v[f"s1-rhat-{t2}-absdiff"] = f"{r.metric_absdiff_median:.3f}"
    v["s1-rhat-frac-min"] = f"{100 * rh.frac_gt_1_1_median.min():.0f}\\%"
    v["s1-rhat-frac-max"] = f"{100 * rh.frac_gt_1_1_median.max():.0f}\\%"
    d = json.load(open(os.path.join(G8, "g8_descriptive.json")))["H5"]
    rk = pd.read_csv(os.path.join(G8, "h5_ranks.csv"), index_col=0)
    v["s1-real-rank-pooled"] = f"{rk.loc['gibbs_warm'].iloc[-1]:.2f}"
    v["s1-real-nmethods"] = str(len(rk))
    for ds, t2 in (("BirdSong", "bird"), ("Lost", "lost"), ("MSRCv2", "msrc")):
        g = d["gibbs_vs_best_baseline"][ds]
        v[f"s1-real-{t2}-diff"] = f3(g["mean_diff"])
        v[f"s1-real-{t2}-wl"] = f"{g['wins']}/{g['losses']}"
        v[f"s1-real-{t2}-rankpos"] = str(d["rank_position"][ds])
    # Lost is a near tie: spread of mean accuracy among the three best methods (oracle excluded). Was typed by hand as
    # "top four within 0.003", which is wrong: the fourth is 0.027 below the first (04/11/2026)
    acc = pd.read_csv(os.path.join(G8, "rq5_acc.csv"), index_col=0).drop(index="oracle")["Lost"]
    top = acc.sort_values(ascending=False).head(3)
    v["s1-real-lost-top3"] = f"{top.iloc[0] - top.iloc[-1]:.3f}"
    m = load_s1(os.path.join(ROOT, "G7", "raw"))
    g = m[m.method == "gibbs_warm"]
    v["s1-cost-median-s"] = f"{g.seconds.median():.0f}"
    v["s1-cost-max-min"] = f"{g.seconds.max() / 60:.1f}"
    v["s1-cost-gpu-gb"] = f"{g.peak_gpu_mb.max() / 1024:.1f}"
    fs = sorted(glob.glob(os.path.join(ROOT, "S2", "raw", "S2T", "s2_tfm_*.csv"))) + \
        [os.path.join(ROOT, "S2", "raw", "S2G", "s2_g_tfm.csv")]
    t = pd.concat([pd.read_csv(f) for f in fs])
    fin = pd.read_csv(os.path.join(ROOT, "S2", "s2_final.csv"))
    t = t[(t.status == "ok") & (t.method == "gibbs_warm") & t.dataset_id.isin(set(fin[fin.kept].did))]
    v["s2-cost-n"] = str(len(t))
    v["s2-cost-median-s"] = f"{t.seconds.median():.0f}"
    v["s2-cost-max-min"] = f"{t.seconds.max() / 60:.1f}"
    v["s2-cost-gpu-gb"] = f"{t.peak_gpu_mb.max() / 1024:.1f}"
    v["s2-cost-ntrain-max"] = f"{int(t.n_train.max()):,}".replace(",", "{,}")
    # Context cap of the sampler (GIBBS["max_context"] = 2000 in icgibbs/g7/methods.py): number of Study 2 datasets whose
    # training set exceeds it, so that some context was subsampled (reviewer 01, item 14)
    nt = t.groupby("dataset_id").n_train.max()
    v["s2-ctxsub-n"] = str(int((nt > 2000).sum()))
    v["s2-ndata"] = str(len(nt))


def s1_collapse_numbers(v):
    """Study 1, PLL q = 0.6: the deficit against CC (tuned) comes from the four datasets with >= 8 classes."""
    t = pd.read_csv(os.path.join(G8, "table_pll_0.6.csv"), index_col=0)
    d = (t.gibbs_warm - t.cc_tun).round(10)
    four = ["mfeat-factors", "semeion", "MiceProtein", "optdigits"]
    rest = d.drop(four)
    v["s1-pll6-rest-diff"] = f3(rest.mean())
    v["s1-pll6-rest-w"] = str(int((rest > 0).sum()))
    v["s1-pll6-rest-n"] = str(len(rest))
    t3 = pd.read_csv(os.path.join(G8, "tabicl_pll_0.6.csv"), index_col=0) if os.path.exists(os.path.join(G8, "tabicl_pll_0.6.csv")) else None
    if t3 is not None and "gibbs_warm" in t3:
        v["s1-tabicl-pll6-four-min"] = f"{t3.loc[four, 'gibbs_warm'].min():.2f}"


def s2e_numbers_and_table(v):
    """exploratory lot S2E (TabICL v2 on Study 1 datasets): MH-corrected blocks and R-hat; appendix table."""
    mh = pd.read_csv(os.path.join(S2, "s2e_mh_summary.csv"))
    rh = pd.read_csv(os.path.join(S2, "s2e_rhat_summary.csv"))
    key = {("small3", "llp", 8.0): "llp", ("small3", "pll", 0.3): "pll3", ("small3", "pll", 0.6): "pll6",
           ("many", "pll", 0.6): "many"}
    for _, r in mh.iterrows():
        k = key[(r.group, r.setting, float(r.level))]
        v[f"s2e-mh-{k}"] = f3(r.mh_minus_approx)
        v[f"s2e-mh-{k}-wtl"] = f"{int(r.mh_better)}/{int(r.ties)}/{int(r.mh_worse)}"
        v[f"s2e-mh-{k}-n"] = str(int(r.n))
        v[f"s2e-mh-{k}-acc"] = f"{r.accept_mean:.2f}"
        v[f"s2e-mh-{k}-llworse"] = f"{-r.logloss_mh_better:.3f}"         # log loss of MH minus default
    cells = pd.read_csv(os.path.join(S2, "s2e_mh_cells.csv"))
    v["s2e-mh-accept-min"] = f"{cells.accept_mean.min():.2f}"
    for _, r in rh.iterrows():
        k = key[(r.group, r.setting, float(r.level))]
        v[f"s2e-rhat-{k}-frac"] = f"{100 * r.frac_gt_1_1_median:.0f}\\%"
        v[f"s2e-rhat-{k}-absdiff"] = f"{r.metric_absdiff_median:.3f}"
    import glob
    ref = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "G7", "raw", "L6", "g7_tabicl_*.csv"))])
    ref = ref[(ref.status == "ok") & (ref.method == "gibbs_warm")].copy()
    ref["level"] = ref.level.astype(float)
    a = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(ROOT, "S2", "raw", "S2E", "s2e_mh*.csv"))])
    a = a[a.status == "ok"].copy()
    a["level"] = a.level.astype(float)
    j = a.merge(ref[["dataset_id", "setting", "level", "seed", "seconds"]], on=["dataset_id", "setting", "level", "seed"],
                suffixes=("_mh", "_approx"))
    ratio = j.seconds_mh / j.seconds_approx
    v["s2e-mh-cost-median"] = f"{ratio.median():.0f}"
    v["s2e-mh-cost-max"] = f"{ratio.max():.0f}"
    nc1 = pd.read_csv(os.path.join(G8, "rhat_summary.csv"))
    nc1["level"] = nc1.level.astype(float)
    lab = {"llp": "LLP, bag 8", "pll3": "PLL, $q=0.3$", "pll6": "PLL, $q=0.6$", "many": "PLL, $q=0.6$, $\\geq$8 classes"}
    out = [r"% generated by paper/scripts/make_tables.py — do not edit", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
           r" & \multicolumn{2}{c}{labels with $\widehat R>1.1$} & \multicolumn{2}{c}{test-metric gap, 2 chains}"
           r" & \multicolumn{2}{c}{MH-corrected $-$ default} \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
           r"Setting (datasets) & TabPFN & TabICL & TabPFN & TabICL & TabPFN & TabICL [acc.\ rate] \\", r"\midrule"]
    abl = pd.read_csv(os.path.join(G8, "ablation_summary.csv"))
    abl["level"] = abl.level.astype(float)
    for (grp, st, lv), k in key.items():
        rr = rh[(rh.group == grp) & (rh.setting == st) & (rh.level == lv)].iloc[0]
        mm = mh[(mh.group == grp) & (mh.setting == st) & (mh.level == lv)].iloc[0]
        if grp == "small3":
            n1 = nc1[(nc1.setting == st) & (nc1.level == lv)].iloc[0]
            pf = (f"{100 * n1.frac_gt_1_1_median:.0f}\\%", f"{n1.metric_absdiff_median:.3f}")
            ab = abl[(abl.variant == "gibbs_warm_mh") & (abl.setting == st) & (abl.level == lv)]
            pm = f3(ab.mean_diff.iloc[0]) if len(ab) else "--"
        else:
            pf, pm = ("--", "--"), "--"
        name = lab[k] + (" (3 small)" if grp == "small3" else f" (4; MH: {int(mm.n_datasets)})")
        out.append(f"{name} & {pf[0]} & {100 * rr.frac_gt_1_1_median:.0f}\\% & {pf[1]} & {rr.metric_absdiff_median:.3f} & "
                   f"{pm} & {f3(mm.mh_minus_approx)} [{mm.accept_mean:.2f}] \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "s2e.tex"), "w").write("\n".join(out) + "\n")


def real_pll_table():
    acc = pd.read_csv(os.path.join(G8, "rq5_acc.csv"), index_col=0)
    rk = pd.read_csv(os.path.join(G8, "h5_ranks.csv"), index_col=0)
    order = [("gibbs_warm", NAMES["gibbs_warm"]), ("hard_em_warm", NAMES["hard_em_warm"]), ("sem_warm", NAMES["sem_warm"]),
             ("naive", NAMES["naive"])] + \
            [(f"{b}_{v}", f"{NAMES[b]} ({'default' if v == 'def' else 'tuned'})")
             for b in ("proden", "glwspll", "rc", "cc") for v in ("def", "tun")] + [("oracle", NAMES["oracle"])]
    ds = ["BirdSong", "Lost", "MSRCv2"]
    best = {d: acc.loc[[m for m, _ in order if m not in ("oracle",)], d].max() for d in ds}
    out = [r"% generated by paper/scripts/make_tables.py — do not edit", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
           r"Method & BirdSong & Lost & MSRCv2 & mean rank \\", r"\midrule"]
    for m, name in order:
        cells = []
        for d in ds:
            x = f"{acc.loc[m, d]:.3f}"
            cells.append(r"\textbf{" + x + "}" if m != "oracle" and abs(acc.loc[m, d] - best[d]) < 1e-12 else x)
        r = f"{rk.loc[m].iloc[-1]:.2f}" if m in rk.index else "--"
        out.append(f"{name} & " + " & ".join(cells) + f" & {r} \\\\")
        if m in ("naive", "cc_tun"):
            out.append(r"\midrule")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "real_pll.tex"), "w").write("\n".join(out) + "\n")


def r4_numbers_and_table(v):
    """Exploratory lot R4 (05/11/2026): every Study 2 cell rerun with a second chain (same start, other sampling seed)."""
    V2R = os.path.join(ROOT, "V2", "results")
    summ = json.load(open(os.path.join(V2R, "v0_summary.json")))["R4"]
    est = json.load(open(os.path.join(V2R, "v0_r4.json")))
    kmap = {"llp_8.0": "llp", "pll_0.3": "pll3", "pll_0.6": "pll6"}
    for k, t in kmap.items():
        a = summ["b_absdiff"][k]
        v[f"r4-{t}-absdiff-med"] = r"\ensuremath{<}0.001" if a["median"] < 0.0005 else f"{a['median']:.3f}"
        v[f"r4-{t}-absdiff-max"] = f"{a['max']:.3f}"
        v[f"r4-{t}-rhat-frac"] = f"{100 * summ['d_rhat'][k]['frac_gt_1_1_median']:.0f}\\%"
        v[f"r4-{t}-flips"] = f"{summ['c_sign_flips'][k]['n_sign_flips']}/{summ['c_sign_flips'][k]['n_datasets']}"
    v["r4-ncells"] = str(summ["n_cells"])
    # chain a: the registered confirmatory analysis; chain b: full-precision test metrics written by the run
    # (pre-written v0_report.py); a and b averaged: from the stored float16 test predictions of both chains (v0_explore E4)
    reg = json.load(open(os.path.join(S2, "s2_confirmatory.json")))
    res = {"a": {"llp": reg["H2_llp8"]["comparator"], "pll3": reg["H4_pll"]["0.3"]["comparator"],
                 "pll6": reg["H4_pll"]["0.6"]["comparator"]},
           "b": {"llp": summ["a_H2prime_chain_b"]["comparator"], "pll3": summ["a_H4prime_chain_b"]["0.3"]["comparator"],
                 "pll6": summ["a_H4prime_chain_b"]["0.6"]["comparator"]},
           "ab": {"llp": est["ab"]["h2_llp"], "pll3": est["ab"]["h4_pll3"], "pll6": est["ab"]["h4_pll6"]}}
    for e, d in res.items():
        for t, c in d.items():
            v[f"r4-{e}-{t}-diff"] = f3(c["mean_diff"])
            v[f"r4-{e}-{t}-p"] = fp(c["p_holm"])
            v[f"r4-{e}-{t}-wtl"] = wtl(c)
    lab = {"llp": "H2$'$: LLP, bag 8, vs DLLP (linear)", "pll3": "H4$'$: PLL, $q=0.3$, vs CC (tuned)",
           "pll6": "H4$'$: PLL, $q=0.6$, vs CC (tuned)"}
    out = [r"% generated by paper/scripts/make_tables.py — do not edit", r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
           r"Test & chain a (registered) & chain b & chains a and b averaged \\", r"\midrule"]
    for t in ("llp", "pll3", "pll6"):
        cells = []
        for e in ("a", "b", "ab"):
            c = res[e][t]
            cells.append(f"{f3(c['mean_diff'])}, {wtl(c)}, {fp(c['p_holm'])}")
        out.append(f"{lab[t]} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "tables", "r4.tex"), "w").write("\n".join(out) + "\n")


def main():
    os.makedirs(os.path.join(PAPER, "tables"), exist_ok=True)
    s1, s2 = load()
    v = numbers(s1, s2)
    extra_numbers(v)
    s2e_numbers_and_table(v)
    s1_collapse_numbers(v)
    r4_numbers_and_table(v)
    write_numbers(v)
    real_pll_table()
    main_table(s1, s2)
    h3_table(s1, s2)
    means_table()
    print(f"{len(v)} numbers; tables written to {os.path.join(PAPER, 'tables')}")


if __name__ == "__main__":
    main()
