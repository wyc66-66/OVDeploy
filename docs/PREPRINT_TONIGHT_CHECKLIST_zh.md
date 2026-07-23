# 今晚 → 明天汇报 → 挂预印：勾选清单

**作者锁定：** Yuechun Wang（王岳纯）· Huaqiao University · 341861408@qq.com  
**GitHub：** https://github.com/wyc66-66/OVDeploy

---

## A. 本地已齐（Agent 已做 / 请核对）

- [x] 英文去匿名：`paper/main_cvpr.tex` 作者块
- [x] Abstract 含 24-backbone 一句
- [x] `paper/advisor_build.pdf` / `d:\ccfa\OVDeploy_preprint_en.pdf`
- [x] 中文预印：`paper/OVDeploy_preprint_zh.pdf` / `d:\ccfa\OVDeploy_preprint_zh.pdf`
- [x] 公开仓 README BibTeX 去 Anonymous + KPI
- [x] endorsement 模板：`docs/templates/arxiv_endorsement_request_en.md`
- [x] 汇报 PPT：`d:\ccfa\开放词汇检测.pptx`
- [x] 汇报 Word：`d:\ccfa\今天汇报的是 OVDeploy.docx`

## B. 今晚你本人必须做

- [ ] 打开 EN PDF 首页确认姓名/单位/邮箱
- [ ] 把 endorsement 模板发给能推荐的人（老师/学长）
- [ ] Push 公开仓（见下方命令）
- [ ] 过一遍 PPT 口播钉：24.8 vs 13.9；OOV 66.4%；不吹 SOTA；不讲 VocabGuard 当 A 主贡献

## C. 明天汇报

- [ ] 带：PPT + Word +（可选）EN/ZH PDF 打印首页
- [ ] 问老师：作者单位表述是否 OK；谁可 arXiv endorsement；是否同意挂预印
- [ ] 学长：对照三层口径 / blocked 诚实 / Line5 放弃人标数字

## D. 汇报后挂 arXiv（仅本人）

1. 登录 https://arxiv.org → Start new submission → **cs.CV**
2. 上传：`advisor_build.pdf`（或源文件包：tex + bib + figures + tables）
3. 等 endorsement 通过 → Submit → 通常 1–2 工作日上线
4. 拿到 arXiv id 后：改 README BibTeX 的 `eprint={TBD}` → 真 id，再 push

## E. 三条命令

```powershell
# 1) Push 公开仓（需已 gh auth / git remote）
cd d:\ccfa\submission-a\ovdeploy-public
git status
git add README.md README_zh.md docs/doat_dense_tables.json docs/EXPERIMENT_TABLE.md
git commit -m "De-anonymize author Yuechun Wang; sync 24-backbone KPI"
git push origin main
```

```powershell
# 2) 重编译英文顾问 PDF
cd d:\ccfa\submission-a
powershell -ExecutionPolicy Bypass -File scripts\compile_paper_advisor.ps1
Copy-Item paper\advisor_build.pdf d:\ccfa\OVDeploy_preprint_en.pdf -Force
```

```powershell
# 3) arXiv 上传用文件（PDF 直传即可）
explorer d:\ccfa\submission-a\paper\advisor_build.pdf
```

## F. 我做不到（勿等 Agent）

- 代你 arXiv endorsement / 上传
- 代你向老师口头要推荐
- 无凭据时 `git push`
