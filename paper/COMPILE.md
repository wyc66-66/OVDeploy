# Compile OVDeploy paper

## One command (official CVPR 2026 template)

```powershell
cd d:\ccfa\论文2
powershell -ExecutionPolicy Bypass -File scripts\compile_paper.ps1
```

Uses official `paper/cvpr.sty` from [cvpr-org/author-kit](https://github.com/cvpr-org/author-kit) with `[review]` mode (line numbers + confidential header).

Output: `paper/main_cvpr.pdf`

## Re-download official files (if needed)

```powershell
python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/cvpr-org/author-kit/main/cvpr.sty', r'paper/cvpr.sty'); urllib.request.urlretrieve('https://raw.githubusercontent.com/cvpr-org/author-kit/main/ieeenat_fullname.bst', r'paper/ieeenat_fullname.bst')"
```

Backup copy: `paper/cvpr_official_2026.sty`, `paper/cvpr_local.sty` (minimal fallback).

## OpenReview

Upload `main_cvpr.tex`, `cvpr.sty`, `references.bib`, `tables/`, `figures/`, `ieeenat_fullname.bst`.
