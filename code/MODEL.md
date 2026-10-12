# 基座模型记录

- 最终提交默认模型：`Qwen/Qwen3-VL-4B-Instruct`
- ModelScope 模型 ID：`Qwen/Qwen3-VL-2B-Instruct`（保留作轻量对照）
- 架构：`Qwen3VLForConditionalGeneration`
- 任务：`image-text-to-text`
- 许可证：Apache-2.0
- 权重文件：`model.safetensors`
- 权重大小：4,255,140,312 bytes
- 权重 SHA-256：`7DE1838C87A5349B016C26A1C3F7D2BC400A3D485F95EF39A7059FFD734977A0`
- `config.json` SHA-256：`BEC4B3D446EFA05807365C9E1CEC03AC590836879D02F3A6DA879971154BDD3B`
- 推理精度：BF16
- 注意力实现：PyTorch SDPA
- 权重提交策略：不打入赛事压缩包；运行前通过 `python code/run_pipeline.py download` 下载并核对上述哈希。

测试环境：Python 3.12.14、PyTorch 2.11.0+cu128、Torchvision 0.26.0+cu128、Transformers 5.18.0、ModelScope 1.40.1、NVIDIA RTX 5070 Ti Laptop GPU 12GB。

## 4B 最终提交记录

- ModelScope 模型 ID：`Qwen/Qwen3-VL-4B-Instruct`
- 权重分片：`model-00001-of-00002.safetensors`（4,967,229,296 bytes，SHA-256 `30A01A0556622645A3CCE87B655BBBBBC1F170C196099F1B666C93202C3339A9`）；`model-00002-of-00002.safetensors`（3,908,490,048 bytes，SHA-256 `046296A2A387EFB43B0C997D5833C789604D168834F6E0D3064BF7BB13D002A6`）。
- 历史冒烟测试：RTX 5070 Ti Laptop 12GB，BF16、`device_map=auto`、PyTorch SDPA 加载成功，无 OOM；当时记录加载约 6 秒、单图推理约 2--3 秒。该历史耗时不等于当前桥梁双阶段多视图流程的逐图总耗时。
- 同一批固定校准图上，4B 的病害识别精度高于 2B 对照；评分字段仍由训练标签分布自动约束。最终提交使用 4B 的桥梁双阶段初筛、桥梁整图+局部裁剪和轨道多裁剪配置。
- 完整原始响应和指标见 `logs/qwen3_vl_4b_smoke.json`、`logs/qwen3_vl_4b_calibration.json`；权重不进入提交包。

## 2026-10-12 当前实测环境

此次留出评测重新核实的环境为 Windows、Python 3.12.14、PyTorch
2.14.1+cu130、Torchvision 0.29.1+cu130、Transformers 5.19.0、Accelerate
1.15.0、Pillow 12.3.0、Safetensors 0.8.0、ModelScope 1.40.1。上文较早的
2.11/5.18 环境记录保留为历史信息，不代表本次运行环境。

主要已验证依赖固定在 `requirements-verified.txt`，不是完整传递依赖锁。
在新的独立环境中复现时，先安装对应 CUDA 13 的 PyTorch wheel，再安装其余
依赖；不要在正在推理的环境里运行安装或升级命令：

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r code\requirements-verified.txt
```

本次 4B 在约 9.8GB GPU 总占用下完成启动及 12 张桥梁冒烟推理，无显存溢出。
单独的权重加载进度耗时约 20 至 40 秒，进程启动还包括模型哈希、导入及断点
重解析，不能混称为纯加载时间。完整桥梁流程每张图有两轮生成，CPU 图片
预处理及原始响应复核也计入评测总耗时。

四张新增桥梁图的 cProfile 诊断产生 8 次真实模型生成，生成函数累计约
55.9 秒；启动时另有 25 条历史原始响应重解析。该剖析有额外开销，不能当作
稳定吞吐或全测试集 ETA。详细性能文件仅在本地
`runs/prompt_ablation_v3/inference_profile.pstats`。当前环境的推理质量需要
重新做成对留出比较，不把旧环境的官方分数当作新环境的评测结果。

`benchmark_inference_threads.py` 在各类别清单首张训练图上完成了 ABBA 线程
诊断（默认 16、4、4、默认 16），八次原始模型响应与结构化结果在各自图片上
均一致。桥梁逐图耗时分别为 46.943、43.733、45.043、48.768 秒；轨道为
21.177、23.640、22.631、21.557 秒。两张图不足以推广性能结论，且 4 线程
没有跨类别的稳定优势，因此不改变默认线程配置。这个诊断关闭普通二次复核，
不能代替完整策略的准确率或吞吐评测。输出仅在
`runs/thread_benchmark_20261012/report.json`，不改写正式结果。

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\benchmark_inference_threads.py `
  --output runs\thread_benchmark_new\report.json
```
