# SigLIP2 cross-bridge holdout experiment

## Why this baseline

The labels are image-level classes, not boxes or segmentation masks. The public bridge U-Net and rail YOLO implementations therefore need annotation types that this dataset does not provide. `google/siglip2-base-patch16-224` is Apache-2.0 and provides reusable visual features; a small class-balanced linear head can be fit directly from the existing training labels.

## Reproduce

Create an isolated Python 3.12 environment with CUDA-enabled PyTorch and the project dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

Download the model once from Hugging Face:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('google/siglip2-base-patch16-224', local_dir='models/SigLIP2-base-patch16-224')"
```

Run a deterministic, label-balanced 120-image evaluation. The bridge group is fully excluded from the classifier and training-derived lexicon; selected track images are excluded too.

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe code\probe_siglip2.py --sample-size 120
```

The paired prompt ablation is:

```powershell
.\.venv\Scripts\python.exe code\evaluate_holdout.py --sample-size 120
```

## Results on 2026-10-07

The split held out all 1,169 labeled bridge images from 范家坪1号大桥. It balanced 84 bridge and 36 track images by label. Exact-match metrics below are restricted to labels represented by at least four remaining training examples. This leaves 75 bridge and 19 track examples; rare or unseen labels are not counted in this comparison.

| Category | Qwen 4B current prompt | SigLIP2 frozen features + linear head |
| --- | ---: | ---: |
| Bridge exact label | 44.0% (33/75) | 46.7% (35/75) |
| Track exact label | 26.3% (5/19) | 42.1% (8/19) |
| Supported classes combined | 40.4% (38/94) | 45.7% (43/94) |
| Non-healthy recall | 45.9% | 89.2% |

Across all 120 balanced samples, Qwen was exact on 38. SigLIP2 was exact on 43 of the 94 samples whose labels were represented by enough training data; the other 26 samples had rare or unseen labels, mostly combinations or labels confined to one bridge. When those labels are not representable by the supervised head, the comparison does not claim an improvement. On this split the Qwen baseline had no exact matches among those 26 samples either.

The prompt ablation also showed that removing the coarse presence hint alone lowered exact accuracy from 31.7% to 27.5%; the multi-view checklist version scored 20.8%. These two prompt changes are not enabled by default.

## Limitations and decision

- This is one bridge-group holdout plus a label-balanced track sample, not the hidden competition test set or an official score estimate.
- The supervised head cannot emit classes absent from its training subset. The evaluation omits 9 rare bridge and 17 rare track examples, so its 45.7% figure must not be presented as full-label accuracy.
- It predicts labels only; it does not replace Qwen-generated descriptions or ratings.
- The current 56.11-point `result/result.json` and the already submitted package remain unchanged. Do not switch the competition output based on this single small holdout. Validate another held-out bridge and a complete generated candidate before packaging or uploading a new model.

Raw metrics: [`../logs/holdout_evaluation.json`](../logs/holdout_evaluation.json) and [`../logs/siglip2_holdout_probe.json`](../logs/siglip2_holdout_probe.json).
