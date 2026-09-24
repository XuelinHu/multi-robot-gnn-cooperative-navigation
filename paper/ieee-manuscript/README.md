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
bibliography — so the two can be read side by side paragraph by paragraph.
After the abstract and prose revision, the English version occupies six pages
and the Chinese version five pages. Their section order, figures, tables, and
bibliography remain aligned; page breaks differ between languages.

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

Figure 1 now uses `gnn_architecture_aligned_v3.pdf`. Its editable Draw.io,
SVG and PNG versions are generated with
`conda run -n pyg python scripts/build_aligned_architecture.py` from the
repository root. The figure prompt is in `figures/prompts/figure1_aligned_v3.md`;
formula and implementation checks are recorded in
`docs/formula_alignment_2026-09-24.md`. Commit `8c4277a` preserves the version
before these corrections, including the previous Figure 1.

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

## Author and funding metadata

Both files carry the same author block and first-page funding footnote:
six authors, Jingchao Wang marked as corresponding author (`\textsuperscript{*}`
on the name), and the two Henan grants in a `\thanks`. `main-en.tex` and
`main-zh.tex` both need `\IEEEoverridecommandlockouts` for `\thanks` to work
inside `\author`.

Both author blocks follow the bundled template's ordinal prefixes: 1st Xuelin
Hu, 2nd Xiaoqin Fu, 3rd Youjing Fu, 4th Jingchao Wang (corresponding author),
5th Pengming Hu, and 6th Simeng Li. The suffixes are superscripted; these
numbers indicate author order, not affiliation identifiers. Read the two
rows from left to right, then top to bottom.

The Chinese version currently prints author names in their romanized form —
the Chinese characters were not available. Affiliations are in Chinese.
Affiliations list only the institution: Liuzhou Railway Vocational Technical
College, Lanzhou University, or Zhongyuan University of Technology. Department
and secondary-school names are omitted in both languages.

### Why the English author block is a `tabular`

`main-en.tex` builds the author block by hand instead of using IEEEtran's
`\IEEEauthorblockN` / `\and`. This is deliberate — do not "simplify" it back.

In conference mode IEEEtran defines

```latex
\renewcommand{\and}[1][\hfill]{\end{@IEEEauthorhalign}#1\begin{@IEEEauthorhalign}}
```

so `\and` does **not** create a tabular column. It ends one `halign` and starts
another, joining the blocks with stretchy `\hfill`. Each row therefore
distributes its leftover space on its own, and the block centres land at
different x offsets from row to row whenever the two rows have different block
widths. Measured on this paper, the two rows were 181 px apart at 200 dpi —
plainly visible, because the Liuzhou affiliation (530 px) is much wider than
the Zhongyuan one (417 px).

IEEE's own template shows the same effect, just less severely (64 px), because
its placeholder affiliations are all the same width.

Writing the block as a real `tabular` makes the columns genuinely shared across
rows. After the change both columns measure constant to within 2 px across both
rows. `\authname` / `\authaff` wrap IEEEtran's own author styles so the
appearance still matches the class defaults.

## Before submission

- Supply the Chinese characters for the six author names in `main-zh.tex`,
  and confirm the Chinese rendering of the first Henan grant.
- Verify the RVO-style baseline is described as a finite-lattice
  implementation, not a formally verified RVO library (Section IV-B and the
  Limitations discussion both say so — keep that wording).
- The paper reports that the mean-aggregation GNN does **not** beat the
  independent MLP, and that strict collision-free success collapses above
  20 robots. Those are the measured results; do not soften them into a
  superiority claim. The Chinese version must carry the same framing.
- When editing one language, apply the matching edit to the other so the two
  stay isomorphic.
