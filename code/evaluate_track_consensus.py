"""Paired Qwen/adapter evaluation on the two adapters' common untouched holdout."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from apply_track_adapter import _load_report, _predict_adapter
from compare_visual_views import check_content_audit
from evaluate_holdout import (
    _gpu_preflight, _holdout_lexicon, _input_item, _paired_comparison,
    _report_lock, _required_gpu_memory, _validate_record, _write_report,
)
from run_pipeline import load_client, load_config
from track4_agent.data import load_assets
from track4_agent.inference import infer_item
from track4_agent.track_consensus import adapter_decision, checked_review, merge_track_reviews
from track4_agent.vision_classifier import model_identity
from track4_agent.vision_review import candidate_path, digest_json, index_unique, probability, review_context, review_image, sha256_file
from compare_visual_views import label_metrics

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent


def common_holdout(train, reports):
    """Only queries unseen by both gradient training and model selection qualify."""
    by_image = index_unique(train, "image")
    common = set(reports[0]["split"]["test"]) & set(reports[1]["split"]["test"])
    if not common:
        raise ValueError("No common untouched holdout")
    for report in reports:
        if common & (set(report["split"]["fitting"]) | set(report["split"]["validation"])):
            raise ValueError("Common holdout entered fitting or checkpoint selection")
    queries = [by_image[image] for image in sorted(common)]
    if any(row["questionCategory"] != "轨道" for row in queries):
        raise ValueError("Holdout contains a non-track image")
    return queries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, default=CODE / "config" / "qwen3-vl-4b-offload.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(args.adapter) != 2 or args.threads < 1:
        parser.error("Exactly two adapters and positive threads required")
    probability(args.threshold)
    output = candidate_path(ROOT, args.output)
    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "MODELSCOPE_OFFLINE"):
        os.environ[name] = "1"
    import torch
    torch.set_num_threads(args.threads)
    train, _, full_lexicon = load_assets(ROOT)
    loaded = [_load_report(path.resolve()) for path in args.adapter]
    reports, states = [pair[0] for pair in loaded], [pair[1] for pair in loaded]
    if any(report["training_manifest_sha256"] != digest_json(train) for report in reports):
        raise ValueError("Adapter training manifest changed")
    queries = common_holdout(train, reports)
    splits = {str(i): ([row for row in train if row["image"] in
                       set(report["split"]["fitting"] + report["split"]["validation"])], queries)
              for i, report in enumerate(reports)}
    for report in reports:
        audit_path = Path(report["config"]["audit"])
        if sha256_file(audit_path) != report["audit_sha256"]:
            raise ValueError("Training content audit changed")
        check_content_audit(ROOT, train, splits, json.loads(audit_path.read_text(encoding="utf-8")))
    excluded = set(reports[0]["split"]["test"]) | set(reports[1]["split"]["test"])
    training = [row for row in train if row["image"] not in excluded]
    lexicon = _holdout_lexicon(training, full_lexicon)
    # Human review guidance from held-out training images must not enter prompts.
    lexicon.pop("review_guidance", None)
    inputs = [_input_item(row) for row in queries]
    config = load_config(args.config.resolve())
    model_path = ROOT / "models" / "SigLIP2-base-patch16-224"
    identity = model_identity(model_path)
    if any(report["model_identity"] != identity for report in reports):
        raise ValueError("Adapters differ from the visual base model")
    context = review_context(ROOT, config, inputs, [], [], lexicon,
                             min_confidence=args.threshold, scenes=())
    context.update(adapter_report_sha256=[sha256_file(path) for path in args.adapter],
                   adapter_sha256=[r["adapter_sha256"] for r in reports],
                   training_manifest_sha256=digest_json(train), threads=args.threads,
                   sources={p.name: sha256_file(p) for p in
                            (Path(__file__), CODE / "apply_track_adapter.py", CODE / "track4_agent" / "track_consensus.py")})
    started = time.perf_counter()
    raw_dir = ROOT / "data" / "raw"
    with _report_lock(output):
        if output.exists():
            if not args.resume:
                raise ValueError("Report exists; pass --resume")
            report = json.loads(output.read_text(encoding="utf-8"))
            if report["context"] != context:
                raise ValueError("Holdout input/config/model/code changed; use a new path")
        else:
            report = {"kind": "paired untouched-common-holdout track consensus diagnostic",
                      "context": context, "status": "running", "baseline": [], "reviews": [],
                      "adapter_predictions": None, "elapsed_seconds": 0.0}
        if report["adapter_predictions"] is None:
            _gpu_preflight(3072)
            from transformers import AutoModel, AutoProcessor
            base_model = AutoModel.from_pretrained(model_path, dtype=torch.float32,
                                                   local_files_only=True, attn_implementation="sdpa")
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            report["adapter_predictions"] = [_predict_adapter(base_model, processor, inputs, state,
                lexicon["categories"]["轨道"]["allowed_labels"], 8, "cuda") for state in states]
            del base_model, processor, states, loaded
            torch.cuda.empty_cache()
            _write_report(output, report)
        adapters = report["adapter_predictions"]
        if len(adapters) != 2 or any(set(a) != {item["filename"] for item in inputs} for a in adapters):
            raise ValueError("Adapter prediction coverage differs")
        for record, truth in zip(report["baseline"], queries):
            _validate_record(record, truth, lexicon, config, raw_dir)
        if [r["truth"] for r in report["baseline"]] != queries[:len(report["baseline"])]:
            raise ValueError("Baseline checkpoint differs from untouched holdout")
        client = None
        if len(report["baseline"]) < len(queries):
            _gpu_preflight(_required_gpu_memory(config))
            client = load_client(config)
            for truth, item in zip(queries[len(report["baseline"]):], inputs[len(report["baseline"]):]):
                prediction, raw = infer_item(client, item, lexicon, config, raw_dir / item["image"],
                                             raw_dir=raw_dir, allow_review=True)
                report["baseline"].append({"truth": truth, "prediction": prediction, "raw": raw})
                _write_report(output, report)
                print(f"[holdout baseline] {len(report['baseline'])}/{len(queries)}", flush=True)
        baseline = [row["prediction"] for row in report["baseline"]]
        proposals, decisions = [], []
        for row in baseline:
            decision = adapter_decision(row, adapters[0][row["filename"]], adapters[1][row["filename"]], args.threshold)
            decisions.append(decision)
            proposals.append({**row, "defectType": decision["final_label"]})
        proposal_by_name = index_unique(proposals, "filename")
        base_by_name = index_unique(baseline, "filename")
        decision_by_name = index_unique(decisions, "filename")
        selected = [item for item in inputs if proposal_by_name[item["filename"]]["defectType"] !=
                    base_by_name[item["filename"]]["defectType"]]
        reviews = report["reviews"]
        index_unique(reviews, "filename")
        if [r["filename"] for r in reviews] != [item["filename"] for item in selected[:len(reviews)]]:
            raise ValueError("Review checkpoint order/coverage differs")
        for review, item in zip(reviews, selected):
            checked_review(review, item, proposal_by_name[item["filename"]], lexicon)
        if len(reviews) < len(selected) and client is None:
            _gpu_preflight(_required_gpu_memory(config))
            client = load_client(config)
        for item in selected[len(reviews):]:
            name = item["filename"]
            vision = {"defectType": proposal_by_name[name]["defectType"],
                      "confidence": min(decision_by_name[name]["adapter_confidences"])}
            reviews.append(review_image(client, item, base_by_name[name], vision, lexicon, config,
                                        raw_dir / item["image"], raw_dir))
            _write_report(output, report)
            print(f"[holdout review] {len(reviews)}/{len(selected)}", flush=True)
        final, audit = merge_track_reviews(baseline, proposals, inputs, reviews, lexicon, args.threshold)
        candidate_records = [{"truth": truth, "prediction": prediction}
                             for truth, prediction in zip(queries, final, strict=True)]
        report.update(status="complete", proposed_count=len(selected),
                      accepted_count=sum(row["accepted"] for row in audit),
                      baseline_metrics=label_metrics(report["baseline"]),
                      candidate_metrics=label_metrics(candidate_records),
                      comparison=_paired_comparison(report["baseline"], candidate_records),
                      candidate=candidate_records, audit=audit,
                      elapsed_seconds=round(report["elapsed_seconds"] + time.perf_counter() - started, 3))
        # Empty track ratings are metadata conventions, never claimed as severity accuracy.
        for key in ("baseline_metrics", "candidate_metrics"):
            report["comparison"][key] = report[key]
        _write_report(output, report)
        print(json.dumps({k: report[k] for k in
                          ("status", "proposed_count", "accepted_count", "baseline_metrics", "candidate_metrics")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
