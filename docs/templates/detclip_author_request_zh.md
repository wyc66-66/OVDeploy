# DetCLIPv2-T 权重索取邮件（中文）

**收件人：** 姚乐炜（香港科技大学，通讯作者）  
**抄送：** 徐航（华为诺亚方舟实验室）  
**主题：** 请求 DetCLIPv2 Swin-T LVIS 零样本 checkpoint（学术复现）

姚老师、徐老师，您好：

我们在做开放词汇目标检测的**部署评测协议**复现（**OVDeploy**：1,220 条固定 LVIS episode、frozen B0–B5、指标 EpisodicAP + OOV-FP）。已在同一 episode 上完成六套公开系统（YOLO-World S/M、OWL-ViT、GLIP-T、GroundingDINO-T/base）的 GPU 评测。

**DetCLIPv2-T**（CVPR 2023，LVIS minival 40.4 AP）是相关工作覆盖表中的第七套系统。我们已集成 MMDet 推理后端与验证脚本，但公开渠道（GitHub / HuggingFace / ModelScope 等）**未找到官方权重与 config**（截至 2026-07）。

能否请您提供以下文件，用于**学术复现**（仅内部评测、不微调、不对外再分发）：

1. Swin-T **LVIS 零样本**权重（`.pth` 或等价格式）  
2. 对应的 **MMDet 配置文件**（`.py`）

收到后我们的安装步骤：

```bash
python scripts/download_detclip_v2.py \
  --checkpoint /path/to/detclipv2_swin_t.pth \
  --config /path/to/detclipv2_swin_t_lvis.py
python scripts/verify_detclip_v2_setup.py --gpu
```

我们会正确引用 DetCLIPv2 论文。非常感谢！

此致  
[姓名 / 单位]
