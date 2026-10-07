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

## Full-label candidate policy

On 2026-10-08, a second evaluation counted every selected label, including
rare and unseen labels. The frozen head reached bridge exact accuracy of
42.86% on 范家坪1号大桥 and 52.38% on 青树湾1号大桥. For track images, the
atomic multilabel head reached Macro-F1 of 0.642 and 0.478 respectively, but
full legal-combination exact accuracy remained 22.22% and 16.67%.

The only candidate policy currently supported is a conservative bridge
support/bottom review: when Qwen's first pass says `完好`, a non-healthy
SigLIP2 label with confidence at least 0.55 triggers a fresh local-Qwen
review. The earlier direct-label ablation reached 55.95% bridge exact
accuracy on the 84-image 范家坪1号大桥 holdout, versus 39.29% for the Qwen
prompt and 42.86% for SigLIP2 alone. That number is not a score for this
two-stage review policy. It is not enabled for track labels, deck or aerial
images, and it does not overwrite the official result. Reproduce it with:

```powershell
.\.venv\Scripts\python.exe code\predict_vision_test.py --device cpu
.\.venv\Scripts\python.exe code\describe_vision_candidates.py `
  --output runs\vision_hybrid\logs\qwen_reviews.json
.\.venv\Scripts\python.exe code\apply_vision_hybrid.py `
  --reviews runs\vision_hybrid\logs\qwen_reviews.json `
  --output runs\vision_hybrid\result\result.json `
  --audit runs\vision_hybrid\logs\vision_hybrid_audit.json
.\.venv\Scripts\python.exe code\run_pipeline.py validate `
  --result runs\vision_hybrid\result\result.json `
  --report runs\vision_hybrid\logs\validation_report.json
```

The current test result has 39 support/bottom bridge images flagged for
possible review. They are not submission changes by themselves: each must receive a valid local
Qwen review before `apply_vision_hybrid.py` accepts it. No fixed description
or rating template is used, and no test labels are read or written.

Before spending GPU time on the test set, the same policy can be evaluated on
two complete bridge-group holdouts. This fits the visual head inside each fold,
runs the ordinary 4B Qwen baseline, applies the exact 39-image-style trigger,
and then measures the accepted Qwen review. The command is resumable and keeps
all intermediate raw responses under `runs/`:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_vision_review.py `
  --groups 范家坪1号大桥,青树湾1号大桥 `
  --output runs\review_validation_v3\paired_review.json `
  --resume
```

The optional bridge kNN candidate head can be compared with the linear head:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_vision_review.py `
  --bridge-method knn --knn-k 5 `
  --groups 范家坪1号大桥,青树湾1号大桥 `
  --output runs\review_validation_v3\paired_review_knn.json `
  --resume
```

On the seven cached bridge holdouts, kNN improved the auxiliary bridge exact
mean from approximately 47.5% to 51.6%; this is still a candidate-label
metric, not an official score. It must pass the paired Qwen-review evaluation
before its test candidates are used.

The baseline and review scores must be compared on the same images. A visual
head score or a direct-label substitution is not evidence that the Qwen review
policy improves the competition output.

The rating field has a separate deterministic calibration candidate. On the
saved bridge holdout, filling an omitted rating from the training-label mode
raised rating accuracy from about 35.7% to 55.0%; this does not change labels,
descriptions, or healthy/track empty ratings. Generate it in isolation with:

```powershell
.\.venv\Scripts\python.exe -X utf8 code\calibrate_ratings.py `
  --output runs\review_validation_v3\rating_calibrated.json `
  --audit runs\review_validation_v3\rating_calibrated_audit.json
```

An additional five-bridge run on the same day produced bridge exact scores
between 16.67% and 64.29%. This spread is why the candidate policy is kept
isolated and conservative; the frozen head is not treated as a universal
replacement for Qwen.
