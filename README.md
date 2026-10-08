# 赛道四：城市路桥隧边坡结构病害智能巡检

本仓库保存赛道四本地多模态智能体工程：默认使用开源 `Qwen3-VL-4B-Instruct`，对桥梁和轨道图片生成赛事要求的七字段 `result.json`，并提供训练标签审核、校准和可恢复推理工具。

## 快速开始

```powershell
$py = "C:\Users\MR\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\code\run_pipeline.py prepare --archive .\赛题四.zip
& $py .\code\run_pipeline.py download
& $py .\code\run_pipeline.py infer --resume
& $py .\code\run_pipeline.py validate
& $py .\code\run_pipeline.py package
```

最终压缩包固定为根目录的 `track4_submission.tar.gz`，重复运行会覆盖同一个文件。训练图片、模型权重和原始数据压缩包不进入 Git；具体模型 ID、哈希和复核记录见 [`code/MODEL.md`](code/MODEL.md)。

## 图片审核

```powershell
& $py .\code\review_server.py
```

打开 `http://127.0.0.1:8765/`，可查看原始标签、评分、4B 校准预测并记录人工意见。人工意见写入 `qa/review_annotations.jsonl`，不会直接改写比赛结果。

## 目录

- `code/`：流水线、模型调用、校验和审核工具
- `data/processed/`：去重后的清单和训练标签词典
- `result/`：当前模型生成的结果和断点
- `design/`：智能体设计书
- `logs/`：推理、校准和验证日志
- `code/OPEN_SOURCE.md`：可复用开源项目和许可证说明

本工程只支持本地开源模型推理，不调用赛事 API，不包含爬虫、测试数据探测或答案硬编码逻辑。

## 优化实验状态

当前代码已加入基于文件名场景的桥梁提示路由（航拍、桥面、墩/支座、梁底）和训练集场景标签统计，并支持桥梁近景多裁剪、候选清单复核及两阶段初筛配置。默认配置为 `code/config/qwen3-vl-4b-review.json`；训练难例审核记录只生成聚合证据提示，不按文件名写入测试答案。

`logs/qwen3_vl_4b_calibration.json`、`logs/calibration_metrics.json` 和 `logs/embedding_probe.json` 记录了本地留出实验。实验结果尚未自动覆盖 `result/result.json`，只有通过本地校准和 `validate` 后才应重新打包提交。

视觉候选升级后，`code/revalidate_visual_candidate.py` 会复核已保存的本地Qwen原始证据；若候选标签被同一响应明确否定，则自动回退到该图片的原模型结果，并保留逐项审计，不重新调用外部服务。

### 2026-10-07 跨桥留出复核

新增 120 张标签均衡留出评测：整座范家坪1号大桥从训练候选和词典中排除，评估其中 84 张桥梁图及 36 张轨道图。Qwen 提示消融显示，删除粗病害初筛提示和增加多视图清单都没有提升，因此正式配置保持不变。Apache-2.0 的 SigLIP2 冻结特征加平衡线性分类头，在 94 张标签训练样本充足的留出图片上精确率为 45.7%，高于 Qwen 同标签口径的 40.4%；另外 26 张罕见/未见标签样本不纳入该比较。目前只有一座桥的独立留出结果。

复现脚本与逐类指标见 [`code/SIGLIP2_EXPERIMENT.md`](code/SIGLIP2_EXPERIMENT.md)。赛事返回的基线反馈为43分（桥梁约4.31/10、轨道约0.43/1）；仓库没有官方评分器，因此本地留出指标只用于版本比较，不能宣称为赛事得分。当前 `result/result.json` 仍由模型和经过审计的视觉候选流程生成；在更多桥梁分组和完整类别输出验证前，不应把探针分类头直接替换为正式模型。
