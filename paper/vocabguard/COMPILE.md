# Compile VocabGuard paper

## One command (official CVPR 2026 template)

```powershell
cd d:\ccfa\submission-b
powershell -ExecutionPolicy Bypass -File scripts\compile_paper.ps1
```

Uses official `paper/cvpr.sty` with `[review]` mode.

Output: `paper/main_cvpr.pdf`

## Prerequisites

1. GPU pipeline: `wsl bash scripts/wsl_run_all_fast.sh`
2. Finalize: `python scripts/finalize_paper.py`

## OpenReview upload

Upload `main_cvpr.tex`, `cvpr.sty`, `references.bib`, `tables/`, `figures/`, `ieeenat_fullname.bst`.
