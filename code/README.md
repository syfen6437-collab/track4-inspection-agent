# 赛道四本地智能巡检首版

本工程默认使用开源多模态模型 `Qwen/Qwen3-VL-4B-Instruct`，在本地完成桥梁与轨道病害识别、病害描述和评定标度生成。推理不调用外部 API，最终输出严格遵循赛事七字段 JSON 结构。

## 快速运行

```powershell
.\code\run.ps1 prepare --archive .\赛题四.zip
.\code\run.ps1 download
.\code\run.ps1 calibrate --sample-size 120
.\code\run.ps1 infer --resume
.\code\run.ps1 validate
.\code\run.ps1 document
.\code\run.ps1 package
```

六小时首版实际校准可先使用24张分层样本；时间充足时再执行计划中的120张：

```powershell
.\code\run.ps1 calibrate --sample-size 24
```

也可运行完整流水线：

```powershell
.\code\run.ps1 all --archive .\赛题四.zip --calibration-sample-size 120
```

当前最终配置为4B桥梁双阶段初筛、桥梁整图加四象限细节拼图，轨道使用整图加四象限细节拼图。输出上限为96个新token；轨道低置信度结果仍可自动复核。默认入口配置为 `code/config/qwen3-vl-4b-review.json`。

完整推理后可用冻结的本地视觉头生成隔离候选。桥梁只对支座/梁底场景、轨道使用原子组合头；每个标签变化样本再次由本地Qwen生成描述和评分，只有合法且与视觉候选一致时才接受：

```powershell
.\.venv\Scripts\python.exe -X utf8 code\apply_visual_candidate.py `
  --vision runs\review_validation_v4\vision_test_knn.json `
  --output runs\visual_candidate_v5\result\result.json `
  --audit runs\visual_candidate_v5\logs\audit.json `
  --descriptions runs\visual_candidate_v5\logs\descriptions.json `
  --track --resume
.\.venv\Scripts\python.exe -X utf8 code\calibrate_ratings.py `
  --input runs\visual_candidate_v5\result\result.json `
  --output runs\visual_candidate_v5\result\result_calibrated.json `
  --audit runs\visual_candidate_v5\logs\rating_audit.json
```

已有候选的原始Qwen响应可在不重新调用模型的情况下重新执行证据矛盾校验；
例如把“候选为粉红色色斑、描述却为无粉红色色斑”的记录自动回退到原结果：

```powershell
& .\.venv\Scripts\python.exe -X utf8 code\revalidate_visual_candidate.py `
  --input runs\visual_candidate_v5\result\result_calibrated.json `
  --audit runs\visual_candidate_v5\logs\audit.json `
  --output runs\visual_candidate_v6\result\result.json `
  --output-audit runs\visual_candidate_v6\logs\revalidation.json
```

候选输出、Qwen原始响应和审计记录始终位于 `runs/`，不会被该命令写入正式结果。

## 训练图片浏览与人工审核

启动一个只读图片浏览器，并把人工意见单独保存到 `qa/review_annotations.jsonl`：

```powershell
$py = "C:\Users\MR\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $py .\code\review_server.py
```

浏览器打开后可按桥梁/轨道、原始标签、文件名、模型低置信度和模型错例筛选。页面显示训练 JSON 的原始 `defectType`、`defectDescription`、`ratingScale(1-5)`，如果存在 `logs/calibration_records.json` 还会显示固定校准集的模型预测。点击保存只产生 QA 记录，不会改写 `data/processed/train_manifest.jsonl`，也不会改写竞赛用的 `result/result.json`。

审核记录可以作为下一轮提示词和标签词典优化依据；正式提交的测试预测仍必须由模型自动生成，不能把人工意见写回测试结果。

开源复用和许可证说明见 [`OPEN_SOURCE.md`](OPEN_SOURCE.md)。当前优先推荐研究 [Label Studio](https://github.com/HumanSignal/label-studio) 和 [FiftyOne](https://github.com/voxel51/fiftyone)；只有新增病害框/掩膜标注时才考虑 [CVAT](https://github.com/cvat-ai/cvat) 或 YOLO。

## SigLIP2 候选复核

`code/evaluate_vision_heads.py` 可在整桥留出集上评估冻结的
`google/siglip2-base-patch16-224` 特征头，`code/predict_vision_test.py` 生成
测试图候选标签，`code/describe_vision_candidates.py` 只对支座/梁底候选图让本地
Qwen 重新生成七字段内部结果，最后由 `code/apply_vision_hybrid.py` 写入独立候选目录。
候选目录与正式 `result/` 隔离，每次改写都有审计日志。完整指标、限制和复现命令见
[`SIGLIP2_EXPERIMENT.md`](SIGLIP2_EXPERIMENT.md)。

桥梁候选头可用 `predict_vision_test.py --bridge-method knn --knn-k 5` 做隔离
对照；轨道仍使用原子多标签头。候选升级前应保留完整留出评估、Qwen原始响应、评分
校准审计和结果校验报告。

正式使用前先运行 `code/evaluate_vision_review.py --prepare-only` 检查两个整桥留出折叠，
再在有空闲 GPU 时运行完整配对评估。只有留出集上复核策略稳定优于基线，才考虑把候选
结果升级为新的提交结果；正式结果升级前仍需运行 `validate` 和 `package`。

桥梁病害评分缺失时，可先用 `code/calibrate_ratings.py` 在 `runs/` 下生成隔离候选；
它只依据训练标签的评分分布补齐默认值，并写出逐项审计，不会覆盖正式结果。

描述缓存会绑定基础结果、视觉候选、配置和筛选参数的哈希。更换其中任一输入后必须使用新的 `--descriptions` 路径；程序会拒绝沿用旧缓存，避免把不同版本的模型响应混入同一候选结果。

## 4B 冒烟测试

4B 模型配置位于 `code/config/qwen3-vl-4b.json`。下载完成后可用同一套入口做小样本测试：

```powershell
& $py .\code\run_pipeline.py --config .\code\config\qwen3-vl-4b.json infer --limit 1
```

该命令会写入当前 `result/` 和 `logs/`，建议先复制首版结果或使用单独工作目录；它不上传官网。

`package` 命令固定生成根目录的 `track4_submission.tar.gz`，重复运行会覆盖同一个文件，不会继续产生带时间戳的提交包。旧的时间戳包属于历史产物，已被 Git 忽略。

最终提交包同时携带 `code/review_annotations.jsonl`，其中只记录训练难例审核状态和证据摘要；它不会改写训练清单，也不会覆盖测试结果。

## 目录

- `data/raw`：公开训练集和初赛测试集。
- `data/processed`：去重训练清单、测试清单和训练标签词典。
- `models`：本地基座模型，不进入提交压缩包。
- `result/result.json`：最终预测文件。
- `logs`：原始模型响应、断点和运行指标，便于复核。
- `design/AI智能体设计方案.docx`：智能体方案设计文档。

## 合规说明

- 测试集结果由本地模型和智能体自动生成，代码不包含测试答案。
- 常规后处理执行 JSON 修复、训练标签词典规范化和输入元数据填充；可选的
  SigLIP2 候选复核是独立的本地模型推理步骤，并保留逐条审计记录。
- 基座模型权重不打入提交包，提交代码记录公开模型 ID 和运行依赖。
- 推理可在模型下载后启用离线模式运行。
- `calibrate` 和 `infer` 会在代码内部强制启用 Transformers、Hugging Face Hub 和 ModelScope 离线开关。
- `--offset` 和 `--limit` 仅用于GPU冒烟测试；正式运行不要设置这两个参数。
