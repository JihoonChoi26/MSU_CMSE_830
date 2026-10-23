"""
app.py: Streamlit data story for the CMSE 830 midterm project.

Run from the project folder with:
    python -m streamlit run app.py

All data preparation lives in cis_prep.py, so the app and the analysis
notebook show the same numbers. The app displays aggregates only.
"""

import matplotlib.pyplot as plt
import streamlit as st

import cis_prep as cp

APP_TITLE = "Work Experience and Career Adaptability"
APP_DESCRIPTION = (
    "For career services staff and institutional researchers. Explore which "
    "forms of work experience students hold at HBCUs, HSIs, and four-year public "
    "institutions, and how students' career adaptability compares across those forms."
)

DIMENSION_NOTES = {
    "Concern": "looking ahead and preparing for one's future career",
    "Control": "taking ownership of career decisions",
    "Curiosity": "exploring options and possible future selves",
    "Confidence": "believing one can solve problems and overcome obstacles",
}

# Key findings for the default view (all institution types, cutoff 4).
# These are the author's interpretations from the analysis notebook.
KEY_FINDING_ACCESS = (
    "The mix of work experience differs by institution type. HBCU students report "
    "the highest internship share (36.2%) but also the highest share with no work "
    "experience (21.1%). HSI students are the least likely to hold an internship "
    "(24.9%) and the most likely to hold a job unrelated to their career goals "
    "(36.3%). The association is modest (Cramér's V = 0.14), so institution type "
    "shifts the mix rather than determining it."
)
KEY_FINDING_CAAS = (
    "Career adaptability differs modestly across forms of work experience. Students "
    "with an internship, other WBL, or a career-aligned job score higher on average "
    "(3.88-3.91 on a 1-5 scale) than students with an unaligned job (3.73) or no work "
    "experience (3.76). The differences are small (Cohen's d of about 0.2 relative to "
    "no work experience), and students in unaligned jobs score about the same as "
    "students with no work experience."
)

st.set_page_config(page_title=APP_TITLE, layout="wide")
cp.set_plot_style()


@st.cache_data
def get_data(align_threshold):
    """Load and prepare the data once per cutoff; Streamlit reuses the result."""
    return cp.build_analytic(cp.load_raw(), align_threshold=align_threshold)


def show_figure(fig):
    """Draw a figure in the left 60% of the page so it stays a readable size.

    st.pyplot stretches a figure to the full container width, which makes plots
    oversized on a wide layout. Returns the right-hand column for text."""
    plot_col, text_col = st.columns([3, 2], gap="large")
    with plot_col:
        st.pyplot(fig)
    plt.close(fig)
    return text_col


# ---------------------------------------------------------------------------
# Header and data context
# ---------------------------------------------------------------------------
st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

with st.expander("About the data and definitions"):
    st.markdown(
        """
**Source.** College Internship Study (CIS), Center for Research on College-Workforce
Transitions (CCWT), UW-Madison. Wave-one survey responses collected 2018-2020.
The data are restricted and de-identified, and this app shows aggregates only.

**Sample.** 2,995 students at 11 four-year campuses (HBCU 1,468, HSI 562, four-year
public 965). One row is one student. Two technical colleges are excluded.

**Forms of work experience.** Each student is assigned to one form, in priority order.
1. *Internship.* Took part in an internship in the past 12 months.
2. *Other WBL.* A co-op, apprenticeship, or practicum, without an internship.
3. *Career-aligned job.* A paid non-internship job the student rated at or above the
   cutoff on how well it builds skills for their desired career (1-5 scale).
4. *Unaligned job.* A paid non-internship job rated below the cutoff.
5. *No work experience.* None of the above.

**Career adaptability.** Measured with the Career Adapt-Abilities Scale (CAAS), 24 items
rated from 1 (not strong) to 5 (strongest), averaged into four dimensions.
"""
    )
    for name, note in DIMENSION_NOTES.items():
        st.markdown(f"- **{name}:** {note}")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    view = st.radio("View", ["Access to work experience",
                             "Career adaptability by work form"])
    align_threshold = st.select_slider(
        "Career-aligned job cutoff", options=[3, 4, 5], value=cp.ALIGN_THRESHOLD,
        help="Minimum rating (1-5) for a job to count as career-aligned. "
             "The default is 4 ('very well'). Change it to see how sensitive "
             "the results are to this definition.")

try:
    df = get_data(align_threshold)
except FileNotFoundError:
    st.error(f"Data file not found at `{cp.DATA_PATH}`. "
             "See the README for how authorized users can obtain the data.")
    st.stop()

with st.sidebar:
    inst_choice = st.selectbox("Institution type",
                               ["All"] + list(df["inst_type"].cat.categories))

data = df if inst_choice == "All" else df.loc[df["inst_type"] == inst_choice]
n_unclassified = int(data["work_form"].isna().sum())
scope = "all institution types" if inst_choice == "All" else f"{inst_choice} campuses"
is_default = inst_choice == "All" and align_threshold == 4

# ---------------------------------------------------------------------------
# View 1: who holds which form of work experience
# ---------------------------------------------------------------------------
if view == "Access to work experience":
    options = ["inst_type", "first_gen", "field", "gender", "race"]
    if inst_choice != "All":
        options.remove("inst_type")  # only one level left after filtering
    by = st.sidebar.selectbox("Compare across", options,
                              format_func=lambda c: cp.GROUP_LABELS[c])

    st.subheader(f"Work-experience form by {cp.GROUP_LABELS[by].lower()} ({scope})")
    if is_default and by == "inst_type":
        st.info(f"**Key finding.** {KEY_FINDING_ACCESS}")

    shares, counts = cp.form_shares(data, by)
    shown = counts.sum(axis=1) >= cp.MIN_CELL
    shares, counts = shares.loc[shown], counts.loc[shown]

    text_col = show_figure(cp.plot_form_shares(data, by=by))

    # Evidence-based summary that updates with the current selection
    if len(shares) >= 2:
        spread = (shares.max() - shares.min()).sort_values(ascending=False)
        lines = []
        for form in spread.index[:2]:
            lo, hi = shares[form].idxmin(), shares[form].idxmax()
            lines.append(f"**{form}** ranges from {shares.loc[lo, form]:.1f}% ({lo}) "
                         f"to {shares.loc[hi, form]:.1f}% ({hi}).")
        test = cp.chi_square_test(counts.loc[:, counts.sum() > 0])
        lines.append(f"Association strength: Cramér's V = {test['cramers_v']:.2f} "
                     f"(chi-square p {'< .001' if test['p_value'] < .001 else '= ' + format(test['p_value'], '.3f')}).")
        text_col.markdown("**In this view.** The largest differences across groups are below.\n\n"
                          + "\n".join(f"- {line}" for line in lines))

    table = shares.round(1)
    table.insert(0, "n", counts.sum(axis=1))
    st.dataframe(table)
    st.caption("How to read. Each bar sums to 100% and shows how students in one group "
               f"are distributed across the five forms. Groups with fewer than {cp.MIN_CELL} "
               f"students are hidden. {n_unclassified} student(s) without a classifiable "
               "work form are excluded.")

# ---------------------------------------------------------------------------
# View 2: career adaptability by form of work experience
# ---------------------------------------------------------------------------
else:
    outcome = st.sidebar.selectbox("CAAS score", list(cp.CAAS_LABELS),
                                   format_func=cp.CAAS_LABELS.get)
    split_options = [None, "inst_type", "first_gen", "gender"]
    if inst_choice != "All":
        split_options.remove("inst_type")
    split = st.sidebar.selectbox(
        "Split by", split_options,
        format_func=lambda c: "No split" if c is None else cp.GROUP_LABELS[c])
    forms = st.sidebar.multiselect("Work forms to show", cp.FORM_ORDER,
                                   default=cp.FORM_ORDER)
    full_scale = st.sidebar.checkbox("Show full 1-5 scale", value=False)

    with st.sidebar.expander("What the CAAS dimensions mean"):
        for name, note in DIMENSION_NOTES.items():
            st.markdown(f"**{name}.** {note.capitalize()}.")
        st.markdown("**Overall adaptability.** The average across all items.")

    if not forms:
        st.info("Select at least one work form.")
        st.stop()

    st.subheader(f"{cp.CAAS_LABELS[outcome]} by work-experience form ({scope})")
    if is_default and outcome == "mean_alladaptability" and split is None:
        st.info(f"**Key finding.** {KEY_FINDING_CAAS}")

    text_col = show_figure(cp.plot_caas_by_form(data, outcome, group=split, forms=forms,
                                                xlim=(1, 5) if full_scale else None))

    summary = cp.caas_summary(data, outcome, group=split)
    summary = summary.loc[summary["work_form"].isin(forms)
                          & (summary["n"] >= cp.MIN_CELL)]

    # Evidence-based summary that updates with the current selection
    overall = cp.caas_summary(data, outcome)
    overall = overall.loc[overall["work_form"].isin(forms) & (overall["n"] >= cp.MIN_CELL)]
    if len(overall) >= 2:
        hi = overall.loc[overall["mean"].idxmax()]
        lo = overall.loc[overall["mean"].idxmin()]
        text_col.markdown(
            f"**In this view.** The highest mean is **{hi['work_form']}** "
            f"({hi['mean']:.2f}) and the lowest is **{lo['work_form']}** ({lo['mean']:.2f}), "
            f"a gap of {hi['mean'] - lo['mean']:.2f} points on the 1-5 scale.")

    st.dataframe(summary.drop(columns="se").round(3), hide_index=True)
    st.caption("How to read. Dots are group means and bars are 95% confidence intervals. "
               "Intervals that do not overlap suggest differences larger than chance. "
               + ("" if full_scale else "The x-axis is zoomed, so use the full-scale "
                  "option to judge the size of the gaps. ")
               + f"Cells with fewer than {cp.MIN_CELL} students are hidden.")

# ---------------------------------------------------------------------------
# Limitations and next steps (visible in both views)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Limitations and next steps")
st.warning(
    "- **Descriptive, not causal.** These are cross-sectional comparisons. Students "
    "who are already more adaptable may be more likely to seek internships or "
    "career-aligned jobs.\n"
    "- **Survey timing overlaps institution type.** All 884 students surveyed in "
    "spring 2020, when the pandemic disrupted internships, attend HBCUs, which is "
    "60% of the HBCU sample.\n"
    "- **Job groups depend on a definition.** The aligned and unaligned job groups "
    "change with the cutoff. A cutoff of 5 sharpens the contrast.\n"
    "- **Self-reported measures.** Work experience and adaptability are both survey "
    "responses.\n"
    "- **Next step.** Link wave one to later waves to examine whether these "
    "differences persist over time."
)