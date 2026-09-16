# IEEE Conference Manuscripts

Two separate papers sharing one IEEE conference template (`IEEEtran.cls`,
template version 6/27/2024, taken unmodified from
`../IEEE-conference-template-062824/`):

| File | Version | Purpose |
| --- | --- | --- |
| `main-en.tex` / `main-en.pdf` | English, 6 pages | Submission copy. |
| `main-zh.tex` / `main-zh.pdf` | Chinese, 5 pages | Author's verification copy, checked against the frozen results. |

Both are limited to six pages. They are deliberately **isomorphic** — identical
section order, the same three tables, the same five figure groups, the same
bibliography — so the two can be read side by side paragraph by paragraph. The
Chinese version runs one page shorter because Chinese text is more compact per
line; no content was dropped to achieve it.

## Build

```bash
bash build.sh          # both
bash build.sh en       # English only
bash build.sh zh       # Chinese only
```

`build.sh` runs xelatex → bibtex → xelatex → xelatex, fails on any LaTeX error,
and warns if a document exceeds six pages.

**XeLaTeX is required for both.** The Chinese version needs `xeCJK`; and under
XeLaTeX, `IEEEtran`'s Times font request falls back **silently** to Latin
Modern, which would fail IEEE's formatting requirement. Both files therefore
set `\setmainfont{Times New Roman}` explicitly — verify with
`pdffonts main-en.pdf`, which should show `TimesNewRoman` and never `LMRoman`.

The Chinese version resolves SimSun / SimHei / FangSong by name. On a machine
without those fonts, change the `\setCJK*font` lines at the top of
`main-zh.tex`.

## Files

| File | Purpose |
| --- | --- |
| `main-en.tex`, `main-zh.tex` | The two manuscripts. |
| `references.bib` | Bibliography, copied from the repository root. |
| `IEEEtran.cls` | Class file, copied from the template archive. |

Figures are not copied here; `\graphicspath{{../../figures/generated/}}` reads
them straight from the repository's generated-figure directory, so regenerating
a figure and rebuilding is enough to update both papers.

## Relationship to the other manuscripts

`../manuscript/` holds an earlier Chinese draft (`paper_overall.tex`,
`draft.md`) using `ctexart`, single-column with no page limit. That draft is a
development document; `main-zh.tex` here is the six-page, IEEE-formatted
rendition and supersedes it for anything being submitted.

## Where the numbers come from

`../aggregate_for_paper.py` reads the frozen evaluation files and prints the
per-cell aggregates used in the tables:

| Paper element | Source |
| --- | --- |
| Table I (main comparison) | `results/baselines_final.csv` |
| Table II (learned models) | `results/extended_learned_physical.csv` |
| Table III (multi-scale) | `results/multiscale_models_test_safety.csv` |
| Graph and safety ablations | `results/ablation_graph_final.csv`, `results/ablation_safety_final.csv` |
| Communication radius | `results/ablation_radius_final.csv` |
| Seed stability | `results/training_seed_comparison.csv` |

Two distance thresholds are reported separately throughout and are never
merged: a **safety violation** at 0.72 m and a **physical collision** at
0.56 m (the robot diameter).

## Before submission

- Replace the placeholder author names and affiliations in both files.
- Verify the RVO-style baseline is described as a finite-lattice
  implementation, not a formally verified RVO library (Section IV-B and the
  Limitations discussion both say so — keep that wording).
- The paper reports that the mean-aggregation GNN does **not** beat the
  independent MLP, and that strict collision-free success collapses above
  20 robots. Those are the measured results; do not soften them into a
  superiority claim. The Chinese version must carry the same framing.
- When editing one language, apply the matching edit to the other so the two
  stay isomorphic.
