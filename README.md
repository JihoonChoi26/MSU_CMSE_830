# Work Experience and Career Adaptability

CMSE 830 midterm project. An interactive Streamlit data story about how college students' work experience relates to their career adaptability.

## Purpose and audience

College students gain work experience in different forms, from internships to paid jobs that may or may not relate to their career goals. These forms are often treated as interchangeable, yet they may not build the same career resources, and access to them may differ across institutions.

The app is for **career services staff and institutional researchers** who design work-based learning programs. It addresses one question.

> Among students at HBCUs, HSIs, and four-year public institutions, how do forms of work experience differ in who holds them and in students' career adaptability?

Users can compare how students are distributed across five forms of work experience, filter by institution type, switch among career adaptability dimensions, and check how results change with the definition of a career-aligned job.

## Data

- **Source.** College Internship Trajectory, Educational Longitudinal Study, Wave-one survey responses.
- **Analytic sample.** 2,995 students at 11 four-year campuses (HBCU 1,468, HSI 562, four-year public 965). One row is one student. Two technical colleges are excluded.
- **Key variables.** Five-level form of work experience (derived), career adaptability (Career Adapt-Abilities Scale, four dimensions plus an overall score on a 1-5 scale), institution type, first-generation status, field of study, and gender.

### Data access

The CIS data are restricted and de-identified, and **they are not included in this repository**. `.gitignore` excludes the `data/` folder and all CSV files. Access is limited to authorized users. [Add the contact for access requests.]

An authorized user places the data file at

```
data/cis.csv
```

or points to it with an environment variable.

```bash
export CIS_DATA_PATH=/path/to/file.csv
```

The app and notebook display aggregate results only, and groups with fewer than 10 students are hidden.

## Repository structure

```
├── README.md
├── app_streamlit.py                  Streamlit app
├── cis_prep.py             data loading, preparation, summaries, and plots (shared by app and notebook)
├── requirements.txt
├── .streamlit/config.toml  keeps the app local to this computer
├── notebooks/
│   └── analysis.ipynb      data-quality checks and exploratory analysis
├── make_slide_figures.py   regenerates the figures used in the slides
└── figures/                slide figures and app screenshot
```

## Setup and run

```bash
pip install -r requirements.txt
python -m streamlit run app_streamlit.py
```

The app opens at `http://localhost:8501`. It runs locally only because the underlying data are restricted.

## Key data decisions

- **Sample.** Wave one only, with the two technical colleges excluded, giving 11 four-year campuses.
- **Forms of work experience.** Each student is assigned to one form in priority order (internship, other work-based learning, career-aligned job, unaligned job, no work experience).
- **Career-aligned cutoff.** A job counts as career-aligned when the student rated it 4 or 5 on how well it builds skills for their desired career. The app lets users switch the cutoff to 3 or 5.
- **Missing data.** Complete-case analysis. Missing job-alignment answers are almost all structural (students without a job were not asked), and only 5 students (0.17%) cannot be classified.
- **Unused variable.** Parents' income is excluded because collection rounds used two incompatible answer scales.

Details and evidence for each decision are in `notebooks/analysis.ipynb`.

## Key limitations

- **Descriptive, not causal.** The comparisons are cross-sectional. Students who are already more adaptable may be more likely to seek internships or career-aligned jobs.
- **Survey timing overlaps institution type.** All 884 students surveyed in spring 2020, when the pandemic disrupted internships, attend HBCUs (60% of the HBCU sample).
- **Definition dependence.** The aligned and unaligned job groups change with the alignment cutoff.
- **Self-report.** Work experience and adaptability are both survey responses.

**Next step.** Link wave one to the later waves to examine whether these differences persist over time.

## AI-use disclosure

- **Tool.** Claude Opus 5.5 (Anthropic).
- **Used for.** Writing and debugging Python code (data preparation module, data-quality checks, plots, Streamlit app, figure script), assembling the analysis notebook from my analysis cells, and drafting text for the README, the app, and example written responses that I then revised.
- **How I checked it.** I ran all code locally, confirmed that the analytic sample matches the parent study's documented count (2,995), recomputed career adaptability scores against the stored values, and checked reported numbers against the notebook output.
- **My decisions.** The research question, audience, sample restriction, alignment cutoff, missing-data approach, and final interpretations.
