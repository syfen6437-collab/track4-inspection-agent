# 可复用开源项目与许可证边界

本工程的推理核心是公开的 `Qwen/Qwen3-VL-Instruct` 系列模型。图片审核工具使用标准 Python HTTP 服务和浏览器原生能力，避免在比赛环境中增加不必要的服务依赖。以下项目已做过调研，可按需要安装或替换当前轻量工具。

## 优先复用

- [Label Studio](https://github.com/HumanSignal/label-studio)，Apache-2.0：适合整图分类、原始标签展示、人工审核和导出 JSON。它最符合当前训练数据已有“每张图一个标签”的形态。
- [FiftyOne](https://github.com/voxel51/fiftyone)，Apache-2.0：适合按模型置信度、错例和相似图片浏览。后续如果要做图像嵌入检索，可以用它替换当前列表页。

## 暂不优先

- [CVAT](https://github.com/cvat-ai/cvat)，MIT：适合目标框、视频和分割标注。当前训练集没有病害框或掩膜，直接引入会增加标注工作量；只有要训练 YOLO 定位器时才值得使用。
- [LabelMe](https://github.com/wkentaro/labelme)，GPL-3.0：适合轻量框/多边形标注，但许可证和当前提交工程的复用边界不如 Apache-2.0/MIT 项目清晰。

## 当前实现的原因

本机没有预装上述大型标注平台，且比赛提交环境需要离线运行。`code/review_server.py` 是项目内的零第三方依赖审核页：它读取训练清单和已有校准记录，只把人工意见写入 `qa/review_annotations.jsonl`，不修改 `train_manifest.jsonl`，也不修改测试集 `result.json`。因此它不会把人工判断伪装成模型推理结果。

如果后续安装 Label Studio 或 FiftyOne，应把它们当作浏览/审核前端，导出的审核文件仍需经过代码审查；不要把人工确认标签直接覆盖到正式测试输出。

## YOLO 的适用边界

当前标签是整图 `defectType`，没有目标框坐标，不能直接训练有意义的 YOLO 检测器。推荐顺序是：先用审核工具为少量难例补充病害区域框，再用 YOLO/RT-DETR 做定位，最后把裁剪区域交给 Qwen-VL 做病害类型和描述。若只需要整图分类，YOLO 不会解决当前细粒度标签混淆问题。
