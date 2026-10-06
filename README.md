# 赛道四：城市路桥隧边坡结构病害智能巡检

本仓库保存赛道四本地多模态智能体工程：使用开源 Qwen3-VL 模型，对桥梁和轨道图片生成赛事要求的七字段 `result.json`，并提供训练标签审核、校准和可恢复推理工具。

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
