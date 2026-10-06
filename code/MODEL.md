# 基座模型记录

- ModelScope 模型 ID：`Qwen/Qwen3-VL-2B-Instruct`
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

## 4B 复核记录

- ModelScope 模型 ID：`Qwen/Qwen3-VL-4B-Instruct`
- 权重分片：`model-00001-of-00002.safetensors`（4,967,229,296 bytes，SHA-256 `30A01A0556622645A3CCE87B655BBBBBC1F170C196099F1B666C93202C3339A9`）；`model-00002-of-00002.safetensors`（3,908,490,048 bytes，SHA-256 `046296A2A387EFB43B0C997D5833C789604D168834F6E0D3064BF7BB13D002A6`）。
- RTX 5070 Ti Laptop 12GB：BF16、`device_map=auto`、PyTorch SDPA 加载成功，无 OOM；加载约 6 秒，单图推理约 2--3 秒。
- 同一批 24 张固定校准图：`defectType` 精确率 12.5%、原子病害 Macro-F1 0.130952、评分准确率 33.33%。2B 对照为 4.17%、0.093268、79.17%。4B 暂不直接替换首版提交，因为评分字段需要单独校准。
- 完整原始响应和指标见 `logs/qwen3_vl_4b_smoke.json`、`logs/qwen3_vl_4b_calibration.json`；权重不进入提交包。
