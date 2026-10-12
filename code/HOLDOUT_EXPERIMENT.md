# Qwen 4B isolated prompt evaluation

This runner evaluates public training images only. It never writes to the
registered `result/result.json`, submission package or design document.
The current official baseline remains 56.11 points; local metrics are not an
official score prediction.

## Controlled variants

| Variant | Prompt label scope | Review replacement |
| --- | --- | --- |
| `baseline` | Existing scene candidates | Historical policy |
| `all_labels` | All legal training labels | Historical policy |
| `strict_review` | Existing scene candidates | Valid review with confidence at least the configured threshold |
| `full_label_scope` | All legal training labels | Strict policy |

All other settings remain fixed. Bridge review permission follows the deployed
`bridge_review_enabled` setting, not `calibration_review`. Scene-specific
triggers are evaluated by the same `infer_item` implementation as deployment.
The baseline config disables bridge review; these selection experiments do
not secretly enable it. Strict selection alone cannot help when no review was
triggered.

`review_selection` acts only after model responses are generated. The runner
therefore replays the earlier variant's identical raw responses for selection-only
comparisons. It runs the same parser and inference decision path, with no new
model calls. Changing label scope or any other config value prevents replay.
Each reused record names its source variant, and resume validates that source.
For the four variants above, 480 evaluated records per fold require only 240
image inference passes (each pass may contain multiple model rounds).

## Prepare Without GPU

Run from the repository root using the project's environment:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_holdout.py `
  --bridge-group 范家坪1号大桥 --sample-size 120 `
  --output runs\prompt_ablation_v3\fanjaping.json --prepare-only
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_holdout.py `
  --bridge-group 青树湾1号大桥 --sample-size 120 `
  --output runs\prompt_ablation_v3\qingshuwan.json --prepare-only
```

Each fold selects 84 label-balanced bridge images and 36 label-balanced track
images. The entire named bridge is excluded from fitting statistics and
descriptions, as are the selected track images. The public label ontology is
retained even for labels absent from the remaining training subset. Holdout
target labels and descriptions never enter model input metadata. Aggregated
human review notes are not included in the holdout lexicon, to avoid leaking
notes from held-out training examples.

Preparation verifies image/model readability by hashing their contents and
records config, split, training manifest, fold lexicon, tokenizer/processor,
weights, implementation and dependency versions. Hashing the 4B weights takes
time but does not load them onto the GPU. `prompt_label_coverage` shows how
many held-out training targets are selectable in each prompt, not how many
predictions are correct.

## GPU Run and Resume

Check that GPU 0 has at least 10,240 MiB free. This is a startup guard, not a
guarantee against out-of-memory errors. The runner refuses to load Qwen while
another large job occupies the GPU; it does not stop other projects' processes.

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_holdout.py `
  --bridge-group 范家坪1号大桥 --sample-size 120 `
  --output runs\prompt_ablation_v3\fanjaping.json --resume --max-images 12
```

The optional limit pauses after 12 newly evaluated records, including replayed
records. It is a smoke-test budget, not a complete accuracy experiment. Resume
without the limit to finish:

```powershell
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_holdout.py `
  --bridge-group 范家坪1号大桥 --sample-size 120 `
  --output runs\prompt_ablation_v3\fanjaping.json --resume
.\.venv\Scripts\python.exe -u -X utf8 code\evaluate_holdout.py `
  --bridge-group 青树湾1号大桥 --sample-size 120 `
  --output runs\prompt_ablation_v3\qingshuwan.json --resume
```

Every completed image is saved with its raw model rounds. Snapshots are flushed
and atomically replaced; a file lock prevents concurrent writers. A failed model
call leaves earlier records intact. Existing output requires `--resume`; changed
inputs, model files, code or configuration require a new output path. Old reports
without a bound context cannot be silently imported as a current experiment.
Resume reconstructs predictions from raw responses on CPU before loading Qwen.
A complete cached run needs no GPU.

## Read the Report

Check `status == complete` and 120 records per variant before comparing folds.
`split`, `variant_order`, `results`, `metrics` and `records` remain available
for existing analysis scripts. Additional fields include:

- `prompt_label_coverage`: included/excluded target labels by category;
- `category_metrics`: exact counts, rating counts, atomic Macro-F1 and healthy/nonhealthy errors;
- `scene_accuracy`: correct counts for aerial, support, bottom and other scenes;
- `model_inferred_images` / `raw_replay_images`: actual inference versus response reuse;
- `comparisons`: paired sample count, gained/lost correct labels and changed image IDs.

Comparisons align image IDs and use only the intersection of completed records.
Partial comparisons are explicitly smaller than the full sample. A single
combined accuracy can hide a bridge/track regression, and label-balanced samples
do not have the hidden test set's class distribution. The first named variant
is the comparison anchor; keep `baseline` first for the intended experiment.

The runner has no automatic promotion or upload operation. Require consistent
evidence across both bridge folds, inspect per-category failures, then generate
a separate model candidate in `runs/` and validate it. The official scorer is
unavailable, so a local improvement still does not establish a higher official score.

## Current Status

The CPU preparation audit measured prompt label coverage as follows. These are
held-out training target counts, not correct predictions or official scores:

| Whole-bridge fold | Baseline bridge targets selectable | Baseline track targets selectable | All-label scope |
| --- | --- | --- | --- |
| 范家坪1号大桥 | 76/84 | 31/36 | 84/84 bridge, 36/36 track |
| 青树湾1号大桥 | 69/84 | 29/36 | 84/84 bridge, 36/36 track |

For example, the first fold's old bridge prompts excluded five expansion-joint
defects and three spalling targets. Coverage alone does not establish that Qwen
can visually identify them. The new runner and existing pipeline passed 51
automated tests at that stage, including changed-input rejection, interrupted
writes, raw response replay and preservation of the official output. The next
update passed 63 tests, including training-only content audits, stricter output
validation and all-label normalization. In all-label mode, the prompt and the
spelling/combination postprocessor now use the same ontology; the default scene
mode is unchanged. Preparation snapshots made before this source change must
not be resumed under the new code; the commands above use `prompt_ablation_v3`.
Training audit instructions are in [`TRAINING_AUDIT.md`](TRAINING_AUDIT.md).

On 2026-10-12, inspection of the registered baseline's 1,274 raw response records
found zero invalid parsed rounds, zero missing closing JSON braces and zero
ordinary second-pass review rounds. The cached paired holdouts for these two
bridges also contained zero ordinary second-pass reviews in their baseline
records. Tightening review replacement alone does not change those cached
predictions; it is not a demonstrated model improvement.

An actual startup attempt was safely refused with approximately 270 MiB free;
new GPU inference must wait for the separate track-one benchmark to release
the device. Prepared checkpoints and passing engineering tests do not count as
completed model evaluation. No new official score is claimed.
