"""
cis_prep.py

Shared data preparation, summary, and plotting helpers for the CMSE 830
midterm project. Both the HW05 notebook and app.py import from this file,
so every number in the notebook and the app comes from the same code.

Data: College Internship Study (CIS), CCWT, compiled T1-T3 master file.
The file stacks survey waves (one row per student-wave response). This
module keeps one wave (T1 by default), which gives one row per student.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Settings. Each of these is an analytic decision; change them here only.
# ---------------------------------------------------------------------------

# The default path is built from this file's own folder, so it works no
# matter which folder the notebook or app is launched from.
MODULE_DIR = Path(__file__).resolve().parent
DATA_FILE = "cis.csv"
DATA_PATH = Path(os.environ.get("CIS_DATA_PATH", MODULE_DIR / "data" / DATA_FILE))

WAVE = "t1"              # wave kept for the midterm (cross-sectional)
DROP_TECHNICAL = True    # drop the two technical colleges (MATC, MC)
ALIGN_THRESHOLD = 4      # q40_desired_career >= 4 counts as career-aligned
MIN_CELL = 10            # groups smaller than this are not displayed
CAMPUS_SEED = 830        # seed for the masked campus codes

FORM_ORDER = [
    "Internship",
    "Other WBL",
    "Career-aligned job",
    "Unaligned job",
    "No work experience",
]

# Okabe-Ito colors (colorblind-safe), one fixed color per work form
FORM_COLORS = dict(zip(FORM_ORDER, ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#595959"]))

GROUP_COLORS = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
GROUP_MARKERS = ["o", "s", "D", "^", "v", "P"]

CAAS_LABELS = {
    "mean_alladaptability": "Overall adaptability",
    "mean_concern": "Concern",
    "mean_control": "Control",
    "mean_curiosity": "Curiosity",
    "mean_confidence": "Confidence",
}

CAAS_ITEMS = {
    "mean_concern": ["q6_ca01_thinking", "q6_ca02_realizing", "q6_ca03_preparing",
                     "q6_ca04_becoming", "q6_ca05_planning", "q6_ca06_concerned"],
    "mean_control": ["q7_ca07_upbeat", "q7_ca08_decisions", "q7_ca09_responsibility",
                     "q7_ca10_beliefs", "q7_ca11_count_on_me", "q7_ca12_right"],
    "mean_curiosity": ["q8_ca13_exploring", "q8_ca14_looking", "q8_ca15_investigating",
                       "q8_ca16_observing", "q8_ca17_probing", "q8_ca18_curious"],
    "mean_confidence": ["q9_ca19_tasks", "q9_ca20_taking_care", "q9_ca21_learning_skills",
                        "q9_ca22_working_ability", "q9_ca23_obstacles", "q9_ca24_problems"],
}
ALL_ITEMS = [item for items in CAAS_ITEMS.values() for item in items]

INST_TYPE_LABELS = {
    "HBCU": "HBCU",
    "HSI": "HSI",
    "4 year public institution": "Four-year public",
    "Technical College": "Technical college",
}
INST_TYPE_ORDER = ["HBCU", "HSI", "Four-year public", "Technical college"]

# One race label appears in a second spelling in the raw file
RACE_FIXES = {"Black, African American.": "Black or African American"}

# Readable names for the grouping variables (used in tables, plots, and the app)
GROUP_LABELS = {
    "inst_type": "Institution type",
    "field": "Disciplinary field",
    "first_gen": "First-generation status",
    "gender": "Gender",
    "race": "Race/ethnicity",
}


# ---------------------------------------------------------------------------
# Loading and preparation
# ---------------------------------------------------------------------------

def load_raw(path=DATA_PATH):
    """Read the compiled master file with every column as text.

    keep_default_na=False stops pandas from guessing which strings mean
    missing. The file marks missing values with the literal string "NA"
    (and some blanks), so those two are converted to NaN explicitly.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"CIS data file not found at: {path.resolve()}\n"
            f"Current working directory: {Path.cwd()}\n"
            "Fix: put the CSV in a data/ folder next to cis_prep.py, "
            "or pass the full path, e.g. load_raw('/full/path/file.csv')."
        )
    raw = pd.read_csv(path, encoding="latin-1", dtype=str, keep_default_na=False)
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")  # leftover row index
    raw = raw.mask(raw.isin(["NA", ""]))  # "NA" and blanks become NaN
    raw["survey_semester"] = raw["survey_semester"].str.lower()  # "Spring" vs "spring"
    return raw


def mask_campus(institution, seed=CAMPUS_SEED):
    """Replace campus names with codes (Campus A, B, ...) in a shuffled order."""
    names = sorted(institution.dropna().unique())
    order = np.random.default_rng(seed).permutation(len(names))
    mapping = {name: f"Campus {chr(65 + i)}" for name, i in zip(names, order)}
    return institution.map(mapping)


def classify_work_form(df, align_threshold=ALIGN_THRESHOLD):
    """Assign each student to one of five mutually exclusive work forms.

    Rules are applied in priority order (np.select keeps the first match):
      1. Internship         q12_internship == Yes
      2. Other WBL          co-op, apprenticeship, or practicum == Yes
      3. Career-aligned job non-internship job and q40 >= threshold
      4. Unaligned job      non-internship job and q40 < threshold
      5. No work experience none of the above and q36_job == No
    Students who fit no rule (for example, a job with no q40 answer)
    are left as missing.
    """
    internship = df["q12_internship"].eq("Yes")
    other_wbl = df[["q12_coop", "q12_apprenticeship", "q12_practicum"]].eq("Yes").any(axis=1)
    job = df["q36_job"].eq("Yes")
    no_job = df["q36_job"].eq("No")
    alignment = pd.to_numeric(df["q40_desired_career"], errors="coerce")

    conditions = [
        internship,
        other_wbl,
        job & (alignment >= align_threshold),
        job & (alignment < align_threshold),
        no_job,
    ]
    labels = np.select(conditions, FORM_ORDER, default="Unclassified")
    form = pd.Series(labels, index=df.index).replace("Unclassified", np.nan)
    return pd.Categorical(form, categories=FORM_ORDER, ordered=True)


def build_analytic(raw, wave=WAVE, drop_technical=DROP_TECHNICAL,
                   align_threshold=ALIGN_THRESHOLD):
    """Return the analytic data frame: one row per student at one wave."""
    df = raw.loc[raw["survey_time"] == wave].copy()
    if drop_technical:
        df = df.loc[df["institution_type"] != "Technical College"].copy()

    numeric_cols = list(CAAS_LABELS) + ALL_ITEMS + ["q40_desired_career", "q39_hours", "age"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["work_form"] = classify_work_form(df, align_threshold)

    inst_levels = [lvl for lvl in INST_TYPE_ORDER
                   if lvl in df["institution_type"].map(INST_TYPE_LABELS).unique()]
    df["inst_type"] = pd.Categorical(df["institution_type"].map(INST_TYPE_LABELS),
                                     categories=inst_levels, ordered=True)
    df["first_gen"] = df["q57_first_gen"].map({"Yes": "First-generation",
                                               "No": "Continuing-generation"})
    df["field"] = df["nsse_major_char"].str.capitalize()
    df["gender"] = df["q51_gender_survey"]
    df["race"] = df["q53_race_survey"].replace(RACE_FIXES)
    df["campus"] = mask_campus(df["institution"])

    # Keep analysis columns only. Campus names, state, and passcodes are
    # dropped so they cannot appear in any output or in the app.
    keep = (["campus", "inst_type", "field", "first_gen", "gender", "race",
             "work_form", "q12_internship", "q12_coop", "q12_apprenticeship",
             "q12_practicum", "q36_job", "q40_desired_career", "q39_hours", "age"]
            + list(CAAS_LABELS) + ALL_ITEMS)
    return df[keep].reset_index(drop=True)


def combined_group(df, outer, inner, sep=" | "):
    """Label for an inner grouping examined within an outer one,
    for example first-generation status within institution type."""
    return (df[outer].astype(str) + sep + df[inner].astype(str)).where(
        df[outer].notna() & df[inner].notna())


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def form_shares(df, by):
    """Counts and row percentages of work form within each level of `by`.
    Students with a missing work form or missing `by` are excluded."""
    d = df.dropna(subset=[by, "work_form"])
    counts = (pd.crosstab(d[by].astype(str), d["work_form"].astype(str))
              .reindex(columns=FORM_ORDER, fill_value=0))
    counts = counts.loc[counts.sum(axis=1) > 0]
    if isinstance(df[by].dtype, pd.CategoricalDtype):  # keep the defined order
        counts = counts.reindex([c for c in df[by].cat.categories if c in counts.index])
    shares = counts.div(counts.sum(axis=1), axis=0) * 100
    return shares, counts


def chi_square_test(counts):
    """Chi-square test of independence plus Cramer's V for a counts table."""
    chi2, p, dof, expected = stats.chi2_contingency(counts)
    n = counts.to_numpy().sum()
    cramers_v = np.sqrt(chi2 / (n * (min(counts.shape) - 1)))
    return {"chi2": chi2, "dof": dof, "p_value": p, "cramers_v": cramers_v,
            "n": int(n), "min_expected": expected.min()}


def caas_summary(df, outcome="mean_alladaptability", by="work_form", group=None):
    """n, mean, SD, and 95% t-interval of a CAAS score by work form
    (optionally split by a second grouping variable)."""
    keys = [by] if group is None else [group, by]
    d = df.dropna(subset=keys + [outcome])
    out = (d.groupby(keys, observed=True)[outcome]
           .agg(n="count", mean="mean", sd="std").reset_index())
    out["se"] = out["sd"] / np.sqrt(out["n"])
    t_crit = stats.t.ppf(0.975, out["n"] - 1)
    out["ci_low"] = out["mean"] - t_crit * out["se"]
    out["ci_high"] = out["mean"] + t_crit * out["se"]
    return out


def compare_forms(df, outcome="mean_alladaptability"):
    """Omnibus tests of whether a CAAS score differs across work forms."""
    groups = [g[outcome].dropna() for _, g in df.groupby("work_form", observed=True)]
    f_stat, f_p = stats.f_oneway(*groups)
    h_stat, h_p = stats.kruskal(*groups)
    return {"anova_F": f_stat, "anova_p": f_p, "kruskal_H": h_stat, "kruskal_p": h_p}


def cohens_d_vs_reference(df, outcome="mean_alladaptability", reference="No work experience"):
    """Standardized mean difference (pooled SD) of each form against a reference form."""
    ref = df.loc[df["work_form"] == reference, outcome].dropna()
    rows = []
    for form in FORM_ORDER:
        if form == reference:
            continue
        x = df.loc[df["work_form"] == form, outcome].dropna()
        pooled_sd = np.sqrt(((len(x) - 1) * x.var() + (len(ref) - 1) * ref.var())
                            / (len(x) + len(ref) - 2))
        diff = x.mean() - ref.mean()
        rows.append({"work_form": form, "n": len(x), "mean_diff": diff,
                     "cohens_d": diff / pooled_sd})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def set_plot_style():
    """Larger, heavier text and light gridlines for readability."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,   # gridlines behind bars and points
        "legend.frameon": False,
    })


def _text_color(hex_color):
    """Black or white label text, whichever contrasts more with the fill."""
    rgb = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "white" if 1.05 / (lum + 0.05) >= (lum + 0.05) / 0.05 else "black"


def plot_form_shares(df, by="inst_type", min_cell=MIN_CELL, title=None):
    """100% stacked bars: share of students in each work form, per group."""
    shares, counts = form_shares(df, by)
    n = counts.sum(axis=1)
    shares = shares.loc[n >= min_cell]
    n = n.loc[shares.index]

    fig, ax = plt.subplots(figsize=(10, 0.55 * len(shares) + 2.2))
    labels = [f"{g}  (n={n[g]:,})" for g in shares.index]
    left = np.zeros(len(shares))
    for form in FORM_ORDER:
        values = shares[form].to_numpy()
        color = FORM_COLORS[form]
        ax.barh(labels, values, left=left, color=color, edgecolor="white", label=form)
        for y, (start, v) in enumerate(zip(left, values)):
            if v >= 6:  # label only segments wide enough to read
                ax.text(start + v / 2, y, f"{v:.0f}%", ha="center", va="center",
                        color=_text_color(color), fontweight="bold", fontsize=9.5)
        left += values

    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of students (%)")
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(FORM_ORDER), loc="lower center", bbox_to_anchor=(0.5, 1.0),
              fontsize=9.5, handlelength=1.2, columnspacing=1.0)
    if title:
        ax.set_title(title, pad=34)
    fig.tight_layout()
    return fig


def plot_caas_by_form(df, outcome="mean_alladaptability", group=None,
                      forms=FORM_ORDER, min_cell=MIN_CELL, title=None, xlim=None):
    """Mean CAAS score with 95% CI for each work form (optionally split by group).

    xlim=None lets matplotlib zoom to the data. Pass a range such as (1, 5)
    to show the full response scale instead.
    """
    summary = caas_summary(df, outcome, group=group)
    summary = summary.loc[summary["work_form"].isin(forms) & (summary["n"] >= min_cell)]
    forms = [f for f in FORM_ORDER if f in set(summary["work_form"].astype(str))]
    y_pos = {form: i for i, form in enumerate(forms)}

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(forms) + 2.0))
    if summary.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, f"No group has at least {min_cell} students.",
                ha="center", va="center", fontsize=12, fontweight="bold")
        return fig
    if group is None:
        for _, row in summary.iterrows():
            y = y_pos[row["work_form"]]
            color = FORM_COLORS[row["work_form"]]
            ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row["ci_low"]],
                                              [row["ci_high"] - row["mean"]]],
                        fmt="o", color=color, ecolor=color, elinewidth=2.5,
                        capsize=4, markersize=8)
            ax.annotate(f"n={row['n']:,}", (row["ci_high"], y), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9)
    else:
        levels = [lvl for lvl in pd.unique(summary[group].astype(str))]
        if len(levels) > len(GROUP_COLORS):
            raise ValueError(f"Too many levels in {group} to split ({len(levels)}).")
        offsets = np.linspace(-0.25, 0.25, len(levels)) if len(levels) > 1 else [0]
        for lvl, off, color, marker in zip(levels, offsets, GROUP_COLORS, GROUP_MARKERS):
            part = summary.loc[summary[group].astype(str) == lvl]
            ys = [y_pos[f] + off for f in part["work_form"]]
            ax.errorbar(part["mean"], ys,
                        xerr=[part["mean"] - part["ci_low"], part["ci_high"] - part["mean"]],
                        fmt=marker, color=color, ecolor=color, elinewidth=2,
                        capsize=3, markersize=7, label=lvl, linestyle="none")
        ax.legend(title=GROUP_LABELS.get(group, group), loc="lower center",
                  bbox_to_anchor=(0.5, 1.0), ncol=len(levels))

    overall = df.loc[df["work_form"].notna(), outcome].mean()
    ax.axvline(overall, color="#333333", linestyle="--", linewidth=1.2)
    ax.text(overall, -0.6, "sample mean", ha="center", va="bottom", fontsize=9)
    ax.set_yticks(range(len(forms)), forms)
    ax.set_ylim(len(forms) - 0.4, -0.8)
    ax.set_xlabel(f"{CAAS_LABELS[outcome]} (1 = Not strong, 5 = Strongest)")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, pad=30 if group else 10)
    fig.tight_layout()
    return fig
