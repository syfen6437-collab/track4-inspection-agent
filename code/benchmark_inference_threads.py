"""Measure CPU-thread settings on fixed training images, without changing predictions."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from evaluate_holdout import _gpu_preflight, _input_item, _write_report
from run_pipeline import DEFAULT_CONFIG, load_client, load_config
from track4_agent.data import load_assets
from track4_agent.inference import infer_item
from track4_agent.vision_review import candidate_path, digest_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "thread_benchmark" / "report.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Threads must be positive")
    output = candidate_path(ROOT, args.output)
    if output.exists():
        parser.error("Output exists; choose a new output path")
    train, _, lexicon = load_assets(ROOT)
    # Fixed first examples per category, not samples selected for model success/failure.
    items = [_input_item(next(row for row in train if row["questionCategory"] == category))
             for category in ("桥梁", "轨道")]
    config = load_config(args.config.resolve())
    _gpu_preflight(10240)
    client = load_client(config)
    default_threads = client.torch.get_num_threads()
    report = {"kind": "paired CPU-thread performance diagnostic, not accuracy evaluation",
              "config": config, "default_threads": default_threads,
              "comparison_threads": args.threads, "lexicon_sha256": digest_json(lexicon),
              "source_sha256": sha256_file(Path(__file__)),
              "images": {item["image"]: sha256_file(ROOT / "data" / "raw" / item["image"])
                         for item in items}, "records": [], "status": "running"}
    _write_report(output, report)
    for item in items:
        # ABBA limits the impact of a one-time warm-up and a simple ordering effect.
        for phase, threads in enumerate((default_threads, args.threads, args.threads, default_threads)):
            client.torch.set_num_threads(threads)
            client.torch.cuda.synchronize()
            started = time.perf_counter()
            prediction, raw = infer_item(client, item, lexicon, config,
                                         ROOT / "data" / "raw" / item["image"],
                                         raw_dir=ROOT / "data" / "raw", allow_review=False)
            client.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            report["records"].append({"image": item["image"], "category": item["questionCategory"],
                                      "phase": phase, "threads": threads,
                                      "elapsed_seconds": round(elapsed, 3),
                                      "prediction": prediction, "raw": raw})
            _write_report(output, report)
            print(f"[{item['questionCategory']}] phase={phase} threads={threads} elapsed={elapsed:.3f}s", flush=True)
    report["output_consistent"] = all(
        len({digest_json({"prediction": row["prediction"], "raw": row["raw"]})
             for row in report["records"] if row["image"] == item["image"]}) == 1
        for item in items)
    report["status"] = "complete"
    _write_report(output, report)
    print(json.dumps({"status": report["status"], "output_consistent": report["output_consistent"],
                      "timings": [{key: row[key] for key in ("category", "phase", "threads", "elapsed_seconds")}
                                  for row in report["records"]]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
