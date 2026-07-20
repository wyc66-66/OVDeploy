# OVDeploy Adoption Hooks — 学界 / 工业认可路径

> 对齐计划「学长口径 + 眼前一亮」：实验覆盖是底座；认可 = 别人用这把尺子做决策。

## Academia

| Deliverable | Path |
|-------------|------|
| Frozen protocol | [`paper/PROTOCOL.md`](../submission-a/paper/PROTOCOL.md) · public copy `ovdeploy-public/docs/PROTOCOL.md` |
| One-click main table | [`scripts/reproduce_main_table.sh`](../submission-a/scripts/reproduce_main_table.sh) |
| Optional leaderboard row schema | [`schemas/ovdeploy_submission_row.schema.json`](../submission-a/schemas/ovdeploy_submission_row.schema.json) |
| Pitch | Venue = eval/benchmark；不伪装 method SOTA |

## Industry

| Deliverable | Path |
|-------------|------|
| Audit-Lite PoC template | [`outreach-mars/ENTERPRISE_POC_PROPOSAL_TEMPLATE.md`](../outreach-mars/ENTERPRISE_POC_PROPOSAL_TEMPLATE.md) |
| Delivery checklist | [`outreach-mars/ENTERPRISE_POC_DELIVERY_CHECKLIST.md`](../outreach-mars/ENTERPRISE_POC_DELIVERY_CHECKLIST.md) |
| Acceptance story (closure example) | [`outreach-mars/INDUSTRY_ACCEPTANCE_STORY.md`](../outreach-mars/INDUSTRY_ACCEPTANCE_STORY.md) |
| Gate language | 给定 \(V\)：EpisodicAP ≥ τ **且** OOV-FP ≤ ε |

## Shared bar

1. True GPU multi-row green cells (`scripts/_debt_cell_ok.py`)  
2. Honest `checkpoint_blocked` only when weights unavailable  
3. Three sentences: ruler / why broad experiments / industry lacks quantification not deployment  

See also: [`docs/ADVISOR_QA_REBUTTAL.md`](ADVISOR_QA_REBUTTAL.md) §学长口径.
