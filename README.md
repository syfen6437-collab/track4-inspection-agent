# 赛道四：城市路桥隧边坡结构病害智能巡检

本仓库保存赛道四本地多模态智能体工程：默认使用开源 `Qwen3-VL-4B-Instruct`，对桥梁和轨道图片生成赛事要求的七字段 `result.json`，并提供训练标签审核、校准和可恢复推理工具。

## 快速开始

```powershell
$py = "C:\Users\MR\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\code\run_pipeline.py prepare --archive .\赛题四.zip
& $py .\code\run_pipeline.py download
& $py .\code\run_pipeline.py infer --resume --artifact-dir .\runs\my_candidate
& $py .\code\run_pipeline.py validate `
  --result .\runs\my_candidate\result\result.json `
  --report .\runs\my_candidate\logs\validation_report.json
```

上面的推理只生成隔离候选，不覆盖当前 56.11 分基线。候选通过完整留出比较和校验后，才考虑升级正式结果。不要把候选目录与正式 `result/` 混用；修改模型、配置或输入时应换一个候选目录，不能强行沿用旧断点。`package` 只打包正式结果，不会自动选择候选。

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

Git 中的 `d2f6bde` 保存了官方56.11分版本。运行 `restore_scored_baseline.py` 会校验提交哈希后恢复完整结果，并写入提交来源；当前53.06分的视觉候选版本已撤回。候选复核只能写入 `runs/`，不能覆盖正式结果。

### 2026-10-07 跨桥留出复核

新增 120 张标签均衡留出评测：整座范家坪1号大桥从训练候选和词典中排除，评估其中 84 张桥梁图及 36 张轨道图。Qwen 提示消融显示，删除粗病害初筛提示和增加多视图清单都没有提升，因此正式配置保持不变。Apache-2.0 的 SigLIP2 冻结特征加平衡线性分类头，在 94 张标签训练样本充足的留出图片上精确率为 45.7%，高于 Qwen 同标签口径的 40.4%；另外 26 张罕见/未见标签样本不纳入该比较。目前只有一座桥的独立留出结果。

复现脚本与逐类指标见 [`code/SIGLIP2_EXPERIMENT.md`](code/SIGLIP2_EXPERIMENT.md)。Git 中保留的最高官方反馈为56.11分（桥梁6.07/10、轨道0.41/1，2026-10-07）。当前 `result/result.json` 已恢复该版本的完整模型输出，并在 `logs/submission_provenance.json` 中登记来源提交和 SHA-256。2026-10-08 的视觉候选版本得到53.06分（桥梁5.74/10、轨道0.39/1），已撤回。仓库没有官方评分器，因此本地留出指标只用于版本比较，不能宣称为赛事得分。

候选复核只能在 `runs/` 隔离目录运行：Qwen 看不到候选标签和原始答案，证据不足、零置信度、缓存不完整都会拒绝或中止。轨道候选在通过与最终提交完全相同的成对留出验证前不会启用。

### 2026-10-12 评测工程更新

`evaluate_holdout.py` 现在支持逐图原子断点、恢复前原始响应核验、GPU 空闲检查和完整输入/模型/代码哈希。桥梁复核开关与正式推理一致；标签范围和复核替换规则分别做对照，仅改变替换规则时复用相同模型响应。新增按桥梁/轨道、场景及成对增益/退化的指标，以及提示词可选标签覆盖检查。运行方法见 [`code/HOLDOUT_EXPERIMENT.md`](code/HOLDOUT_EXPERIMENT.md)。这些是评测可靠性改进，不代表官方成绩已经提高；56.11 分正式结果保持不变。

训练图片的只读完整解码、重复内容和冲突标签检查见 [`code/TRAINING_AUDIT.md`](code/TRAINING_AUDIT.md)。加强后的结果校验要求七个字段均为字符串，并按文件名核对输入元数据，拒绝重复 JSON 键、非 UTF-8 和非有限数字。完整标签实验的提示与后处理使用同一词典范围，默认场景范围保持不变。审核记录采用原子替换及同服务内并发保护，保存失败不会截断上一份记录。本次代码通过 67 项自动测试。

### 当前提交准备状态

本次工程完善后通过 91 项自动测试。正式结果仍为历史 56.11 分版本，1,274 条输出通过严格校验，保存的模型原始响应重新解析后与正式预测全部一致。没有应用未验证的候选，也没有产生已确认的新官方分数。

设计书已在官方模板内更新并逐页检查。打包改为临时生成、全文件哈希核验后原子替换；中断或读回失败时不覆盖上一份包。提交包内的 `code/logs/submission_manifest.json` 分别记录结果来源提交、当前工程提交和每个文件的 SHA-256。

```powershell
.\code\run.ps1 validate
.\code\run.ps1 package
.\code\run.ps1 package-check
```

唯一当前提交包为 `track4_submission.tar.gz`，不要选旧的带时间戳压缩包。重复上传相同预测不是提分方案。完整实验进度和限制见 [`code/SUBMISSION_STATUS.md`](code/SUBMISSION_STATUS.md)；官网上传及开始测评仍需用户确认。
