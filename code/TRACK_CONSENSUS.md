# 轨道视觉微调与独立复核

这条路径是隔离实验，不是已提分的正式版本。核心生成模型仍为本地开源
`Qwen/Qwen3-VL-4B-Instruct`；两个 `google/siglip2-base-patch16-224`
轨道视觉适配器只提供辅助分类。两者使用不同随机划分，但来自同一公开训练集，
不能把它们的一致意见视为两个统计独立专家。

## 决策规则

1. 两个适配器选择同一合法标签，置信度均达到 0.55。
2. 保留原 Qwen 的“完好”预测；不允许把原有病害改为“完好”。
3. 仅对满足条件的不同标签调用 Qwen 独立复核。提示不包含适配器答案或原答案。
4. Qwen 的五个内部字段齐全、证据不矛盾、置信度达到 0.55，且标签与两个
   适配器一致时，采用 Qwen 自己生成的标签、描述和标度；否则保留原输出。
5. 正式输出只有七字段，桥梁记录不参与这条路径，元数据从输入清单保留。

适配器置信度是合法标签空间内的归一化分数，Qwen 置信度是模型自报值，
都不是经过概率校准的正确率。阈值仅是固定接受条件，不能解读成 55% 的可靠性。

## 已完成的隔离测试复核

首次基线保护候选改变 57 条轨道标签，随后独立 Qwen 实际复核全部 57 张：
53 张输出满足严格结构规则，4 张不合格；其中仅 2 张得到三模型一致支持。
两张的 Qwen 置信度均为 0.9。完整候选为 1,274 条，桥梁 974 条零变化，
七字段校验通过。候选保存在 `runs/track_adapter_qwen_review_20261013/`。
没有测试真值或新官方反馈，因此无法确认这两条改变是否提高成绩；此候选
没有替换正式结果，也没有上传。

首次候选工具已经补充逐图原子断点、恢复前原始响应核验、来源哈希、输出锁和
置信度接受条件。既有报告来自早期工具，不能伪造为新增来源绑定下的断点；
新实验应使用新目录。所有自动后处理规则通过共享 `track_consensus.py` 使用。

## 成对留出验证

`evaluate_track_consensus.py` 只使用两个适配器外层留出的交集，共 66 张公开训练
图片；它们均未进入任一适配器的梯度拟合或内部选轮次验证。原始字节和解码像素
审计确认拟合/评测内容隔离。Qwen 词典排除两个外层留出集的并集，不带人工审核
提示；模型只接收图片和元数据，标签只在推理结束后计算指标。

基线使用 Qwen 正式轨道推理策略，候选采用与测试候选相同的共享接受规则。
报告保留完整基线原始响应、两份适配器预测、独立复核响应和逐项接受理由。
按同一图片计算标签精确匹配、原子 Macro-F1、新增正确/丢失原正确、病害漏检
和完好误报。轨道标度目标全为空，不宣称标度预测能力。

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_track_consensus.py `
  --adapter runs\track_adapter_20261013\report.json `
  --adapter runs\track_adapter_20261014\report.json `
  --config code\config\qwen3-vl-4b-offload.json `
  --output runs\track_consensus_holdout_20261013\report.json
```

存在断点时添加 `--resume`。更换模型、配置、代码或图片后必须使用新目录。
这 66 张已用于探索适配器一致策略，属于开发诊断集，不能当作从未参与过决策的
最终独立验收集；即使有本地收益，也不能声称官方分数提高。

完整对照已完成：基线与候选均为 30/66 精确匹配（45.45%），原子 Macro-F1
均为 0.401684，非完好漏检 9 张、完好误报 6 张。9 条适配器提议的独立
Qwen 回答全部与提议标签不同，最终接受 0 条，新增正确 0、丢失原正确 0。
含准备和模型加载的运行耗时 1,136.001 秒。完整断点 `--resume` 回放成功，
没有加载模型，指标和接受结果一致。该路径未证明提分收益，不登记为正式预测。

基线的 8 张支座锈蚀全部判成破损或完好，6 张完好全部误报；这只说明本诊断
集上的短板，不是测试集逐图答案。新增通用定义提示的随机对照由
`evaluate_track_prompt.py` 运行，排除这 66 张，并同时比较原基线和
`track_definitions`，实验开关 `track_label_definitions` 默认关闭。

32 张随机对照已完整运行，全部排除前述 66 张诊断图片：

| 指标 | 原基线 | 通用定义提示 |
| --- | ---: | ---: |
| 精确匹配 | 12/32（37.50%） | 13/32（40.63%） |
| 原子 Macro-F1 | 0.354757 | 0.390340 |
| 非完好漏检 / 完好误报 | 3 / 6 | 3 / 6 |

成对新增正确 1 张、丢失原正确 0 张、标签变化 3 张。两组累计运行耗时分别
481.420、481.072 秒，不能据此宣称稳定加速。通用定义改善了小样本分类，但
没有足够证据支持替换正式结果；默认开关关闭，提供独立实验配置
`config/qwen3-vl-4b-track-definitions.json`。3,402 条训练元数据的默认提示逐条
与本次改动前源码比较均一致。轨道标度全为空，不作为分级能力指标。

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_track_prompt.py `
  --sample-size 32 --seed 20261015 `
  --exclude-report runs\track_consensus_holdout_20261013\report.json `
  --output runs\track_definitions_holdout_20261013\report.json
```

完整断点可添加 `--resume` 做原始响应回放验证，不重新推理；代码或输入改变时
拒绝续用旧断点。上述 32 张只作为开发验证，需要扩大随机样本并验证正常图、
组合病害和不同场景后，才考虑生成新的正式候选。

## 新候选运行

以下命令只写 `runs/`。适配器中间结果只改分类，描述仍是基线描述，因此不可
直接提交；必须经过 Qwen 完整复核和成对验证，再决定是否登记为正式候选。

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\apply_track_adapter.py `
  --adapter runs\track_adapter_20261013\report.json `
  --adapter runs\track_adapter_20261014\report.json `
  --output runs\track_candidate_new\result.json `
  --audit runs\track_candidate_new\audit.json
.\.venv\Scripts\python.exe -u -X utf8 code\review_track_candidate.py `
  --candidate runs\track_candidate_new\result.json `
  --adapter-audit runs\track_candidate_new\audit.json `
  --output runs\track_review_new\result.json `
  --audit runs\track_review_new\audit.json
```

复核中断后加 `--resume`；完整断点可加 `--resume --revalidate-only`，重新解析
原始文本、重算接受条件并校验，完全不加载 GPU。没有完整断点时拒绝离线重验，
不能把人为改过的结构字段伪装成模型结果。

所有输入/输出路径须不同；候选输出必须在项目 `runs/` 内。正式 `package`
入口只读取已登记的 `result/result.json`，不会自动选择此目录的实验。
