"""Generate tables_official.tex (Tables III/IV) and tables_sequences.tex (Table V) from the official scoring service output."""
import json, pathlib, sys
import pandas as pd
here = pathlib.Path(__file__).resolve().parent
src = here.parent / "official_scores" / "final_kaggle_w_uni.json"
BASE = {"Score": 0.538, "ATE20": 1.261, "AVE": 0.461}            # organizers' baseline, full test set (starter kit)
PLAT = [("car", "Wheeled / Car"), ("human", "Handheld / Human"), ("dog", "Legged / Quadruped"), ("drone", "Aerial / Drone")]

def f(x, nd=3):
    try: return f"{float(x):.{nd}f}"
    except Exception: return "--"

if src.exists():
    d = json.load(open(src))
    ov = pd.DataFrame(d["overall"]["data"], columns=d["overall"]["headers"])
    pl = pd.DataFrame(d["platform"]["data"], columns=d["platform"]["headers"])
    sq = pd.DataFrame(d["seqs"]["data"], columns=d["seqs"]["headers"])
    print("overall columns:", list(ov.columns)); print(ov.to_string(index=False)); print("platform columns:", list(pl.columns)); print(pl.to_string(index=False)); print("seq columns:", list(sq.columns), len(sq))
    # column name resolution (robust to the service's labels)
    def col(df, *keys):
        for c in df.columns:
            if any(k.lower() in str(c).lower() for k in keys): return c
        raise KeyError(keys)
    o = ov.iloc[0]
    ours = {"Score": o[col(ov, "score", "total")], "ATE20": o[col(ov, "ate")], "AVE": o[col(ov, "ave")], "RTE": o[col(ov, "rte")]}
    rows_p = {}
    pc = col(pl, "platform"); 
    for _, r in pl.iterrows():
        name = str(r[pc]).lower()
        for key, _ in PLAT:
            if key in name or (key == "dog" and "quadruped" in name) or (key == "human" and "handheld" in name):
                rows_p[key] = (r[col(pl, "ate")], r[col(pl, "ave")], r[col(pl, "rte")])
    seq_rows = [(int(r[col(sq, "seq")]) if "seq" in " ".join(map(str, sq.columns)).lower() else i + 1, r[col(sq, "ate")], r[col(sq, "ave")], r[col(sq, "rte")]) for i, (_, r) in enumerate(sq.iterrows())]
else:
    ours = {"Score": None, "ATE20": None, "AVE": None, "RTE": None}; rows_p = {}; seq_rows = [(i + 1, None, None, None) for i in range(89)]
    print("official scores not available yet -> placeholders")

t3 = f"""\\begin{{table}}[htbp]
    \\centering\\footnotesize
    \\caption{{Overall challenge performance (official scoring service, full 89-sequence test set). The TartanIMU Score is
    $0.6\\,\\mathrm{{AVE}}/0.7356 + 0.4\\,\\mathrm{{ATE}}_{{20}}/3.1160$, dimensionless, \\emph{{lower is better}} (all-zero = 1.000).
    \\textbf{{RTE does not enter the ranking}}; it is defined in the caption of Table~\\ref{{tab:sequence_results}}.
    Baseline numbers are the organizers' published full-test values.}}
    \\setlength{{\\tabcolsep}}{{3.5pt}}
    \\begin{{tabular}}{{lcccc}}
        \\toprule Method & Score $\\downarrow$ & ATE$_{{20}}$ $\\downarrow$ & AVE $\\downarrow$ & RTE $\\downarrow$ \\\\ \\midrule
        TartanIMU Baseline & {f(BASE['Score'])} & {f(BASE['ATE20'])} & {f(BASE['AVE'])} & -- \\\\
        Ours (Hack2Publish) & {f(ours['Score'], 4)} & {f(ours['ATE20'])} & {f(ours['AVE'])} & {f(ours['RTE'])} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\label{{tab:overall_results}}
\\end{{table}}
"""
lines = []
for key, name in PLAT:
    a, v, r = rows_p.get(key, (None, None, None)); lines.append(f"        {name} & {f(a)} & {f(v)} & {f(r)} \\\\")
avg = ("--", "--", "--")
if rows_p:
    import numpy as np
    avg = tuple(f(np.mean([rows_p[k][i] for k, _ in PLAT if k in rows_p])) for i in range(3))
t4 = f"""\\begin{{table}}[htbp]
    \\centering\\footnotesize
    \\caption{{Performance by platform (official scoring service). RTE is defined in the caption of Table~\\ref{{tab:sequence_results}}.}}
    \\begin{{tabular}}{{lccc}}
        \\toprule Platform & ATE$_{{20}}$ $\\downarrow$ & AVE $\\downarrow$ & RTE $\\downarrow$ \\\\ \\midrule
{chr(10).join(lines)}
        \\midrule Average & {avg[0]} & {avg[1]} & {avg[2]} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\label{{tab:platform_results}}
\\end{{table}}
"""
(here / "tables_official.tex").write_text(t3 + "\n" + t4)
# Table V: 89 sequences in 3 columns of 30
seq_rows = sorted(seq_rows, key=lambda r: r[0])[:89]
while len(seq_rows) < 89: seq_rows.append((len(seq_rows) + 1, None, None, None))
cols = [seq_rows[0:30], seq_rows[30:60], seq_rows[60:89] + [None]]
body = []
for i in range(30):
    cells = []
    for c in cols:
        r = c[i] if i < len(c) else None
        cells.append(f"{r[0]} & {f(r[1])} & {f(r[2])} & {f(r[3])}" if r else " & & & ")
    body.append("        " + " & ".join(cells) + " \\\\")
t5 = f"""\\begin{{table*}}[htbp]
    \\centering\\footnotesize
    \\caption{{Per-sequence challenge results over the 89 test sequences (official per-sequence scoring service; submission md5
    \\texttt{{fd12d4f25d7b52d44aaa4e4a14534378}}). \\textbf{{RTE}} (Relative Trajectory Error, m) follows Sturm \\emph{{et al.}}~\\cite{{sturm2012rgbd}}:
    over sliding windows of $\\Delta t = 5$\\,s, the RMSE of the difference between predicted and ground-truth relative displacement,
    $\\mathrm{{RTE}} = \\sqrt{{\\frac{{1}}{{n}}\\sum_i \\lVert (\\hat{{\\mathbf{{p}}}}_{{i+\\Delta t}} - \\hat{{\\mathbf{{p}}}}_i) - (\\mathbf{{p}}_{{i+\\Delta t}} - \\mathbf{{p}}_i) \\rVert^2}}$.
    Sequence ids follow the scoring service's ordering.}}
    \\label{{tab:sequence_results}}
    \\setlength{{\\tabcolsep}}{{4pt}}
    \\begin{{tabular}}{{cccc @{{\\hspace{{2em}}}} cccc @{{\\hspace{{2em}}}} cccc}}
        \\toprule
        Seq. & ATE$_{{20}}$ & AVE & RTE & Seq. & ATE$_{{20}}$ & AVE & RTE & Seq. & ATE$_{{20}}$ & AVE & RTE \\\\ \\midrule
{chr(10).join(body)}
        \\bottomrule
    \\end{{tabular}}
\\end{{table*}}
"""
(here / "tables_sequences.tex").write_text(t5)
print("tables written")
