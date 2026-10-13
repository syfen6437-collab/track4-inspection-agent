# 赛题四交付与复现说明

当前正式预测保留历史官方反馈 56.11 分的 Qwen3-VL-4B 完整输出。工程增加了
模型实验、断点恢复和审计检查，但新方案尚未证明官方提分。重复上传同一份 JSON
不会改善成绩；不要把工程提交号当成新的模型成绩。

## 官网需要的文件

作品压缩包使用工程根目录唯一的 `track4_submission.tar.gz`。若官网分别要求
预测 JSON，选择 `result/result.json`。压缩包只包含 `code/`、`design/`、`result/`，
没有原始图片和模型权重；代码、配置、依赖说明、模型日志及设计书都已包含。

正式预测应为 1,274 条：桥梁 974、轨道 300，每条为七个字符串字段。
包内 `code/logs/submission_provenance.json` 记录结果来源，
`code/logs/submission_manifest.json` 记录工程提交、设计书和所有文件哈希。
根目录 `logs/package_report.json` 记录整个压缩包的哈希。

官网公开首页于2026年10月13日显示初赛打榜截止10月15日；详细截止时刻、账号
权限和测评次数以登录后的官方通知为准。上传和点击“开始测评”需要用户最后确认。

## 解压后的只读验收

在解压目录打开 PowerShell。已安装工程依赖的 Python 可直接运行以下命令；
仅检查包和回放模型日志时，Python 3.11 以上加 Pillow、python-docx 即可，不要求 CUDA：

```powershell
python -X utf8 code\run_pipeline.py --help
python -X utf8 code\run_pipeline.py package-check --archive '压缩包的实际路径'
python -X utf8 code\replay_submission.py --archive '压缩包的实际路径'
```

回放工具先核对 gzip、目录和全部文件哈希，再重新解析每条原始模型响应并与正式
七字段结果比较。它只支持当前登记的首轮结果策略，遇到额外复核轮次会拒绝。
该检查证明保存结果与日志一致，不能证明图片判断准确，也不能代替重新加载模型
推理或官方评分。工具不写文件，不把结果 JSON 作为生成新答案的来源。

## 新环境复现本地推理

以解压根目录为当前目录。使用独立 Python 3.12 环境、NVIDIA CUDA GPU；
基线设备为 RTX 5070 Ti Laptop 12GB。先建虚拟环境并安装实测主要依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.14.1+cu130 torchvision==0.29.1+cu130 --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r code\requirements-verified.txt
.\code\run.ps1 prepare --archive '官方赛题四.zip的实际路径'
.\code\run.ps1 download
```

首次数据准备需要原始约20GB压缩包及足够的解压空间，模型下载需要网络及约9GB
权重空间。下载后推理入口强制使用本地模型和离线开关，不调用外部推理接口。
依赖文件记录实测主要版本，不是完整传递依赖锁；当前软件环境不保证逐字符复现
历史软件版本产生的结果。模型 ID、许可证及权重哈希见 `MODEL.md`。

不要在已有正式结果上直接运行 `all`。生成独立候选并校验：

```powershell
.\code\run.ps1 infer --resume --artifact-dir .\runs\reproduction
.\code\run.ps1 validate --result .\runs\reproduction\result\result.json --report .\runs\reproduction\logs\validation_report.json
```

12GB 显卡的空闲显存不足时，可显式使用已测试的 CPU 分担配置；速度和输出需要
独立记录，不能当成历史基线重现成功：

```powershell
.\code\run.ps1 --config .\code\config\qwen3-vl-4b-offload.json infer --resume --artifact-dir .\runs\reproduction_offload
```

更换配置、模型、输入或实现后使用新的候选目录，避免旧断点混入。先完成留出
对照，再决定是否整版采用候选；不按测试样本逐条人工修改答案。训练图片审核
入口为 `code/review_server.py`，只保存独立审核意见，不能直接覆盖正式预测。

## 已验证的工程与模型变化

两组轨道视觉适配器提高了标签精确匹配，但病害漏检也增加。它们结合独立 Qwen
复核后，在66张共同留出图上没有完整流程收益。通用轨道定义提示在另一组32张
图上从12张到13张正确，样本太小，默认关闭。完整报告和命令见
`GPU_REFINEMENT.md`、`TRACK_CONSENSUS.md`。不把实验候选升级为正式提交。

本地单元测试主要验证断点、缓存、结构化处理和打包行为，不能代表识别准确率：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s code\tests -p 'test*.py' -q
```

GitHub 工程来源为 https://github.com/syfen6437-collab/track4-inspection-agent 。
原始数据、基座权重、实验权重及压缩包不进入 Git。需要确认当前提交包身份时，
运行回放检查或 `package-check`，读取报告中的工程提交和 SHA-256。
