from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from track4_agent.data import load_assets
from track4_agent.inference import _calibration_sample, _macro_atomic_f1, infer_item
from track4_agent.model import ModelClient


def main() -> None:
    config = json.loads((CODE_DIR / "config" / "qwen3-vl-4b.json").read_text(encoding="utf-8"))
    train, _, lexicon = load_assets(WORKSPACE)
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    sample = _calibration_sample(train, sample_size, int(config["seed"]))
    client = ModelClient(
        WORKSPACE / config["model_dir"],
        max_new_tokens=int(config["max_new_tokens"]),
        global_max_side=int(config["global_max_side"]),
        crop_size=int(config["crop_size"]),
    )
    records = []
    started = time.time()
    for index, row in enumerate(sample, start=1):
        item = {
            "id": row["image"],
            "image": row["image"],
            "questionCategory": row["questionCategory"],
            "bridgeName": row["bridgeName"],
            "defectLocation": row["defectLocation"],
            "filename": row["filename"],
        }
        prediction, raw = infer_item(
            client,
            item,
            lexicon,
            config,
            WORKSPACE / "data" / "raw" / row["image"],
            allow_review=False,
        )
        records.append({"truth": row, "prediction": prediction, "raw": raw})
        print(f"[4b-calibrate] {index}/{len(sample)} {row['questionCategory']} {row['filename']}", flush=True)
    pairs = [(r["truth"]["defectType"], r["prediction"]["defectType"]) for r in records]
    exact = sum(a == b for a, b in pairs)
    rating = sum(r["truth"]["ratingScale(1-5)"] == r["prediction"]["ratingScale(1-5)"] for r in records)
    report = {
        "model": config["model_id"],
        "sample_size": len(records),
        "defect_type_exact_accuracy": round(exact / len(records), 6) if records else 0,
        "atomic_macro_f1": round(_macro_atomic_f1(pairs), 6),
        "rating_accuracy": round(rating / len(records), 6) if records else 0,
        "elapsed_seconds": round(time.time() - started, 3),
        "records": records,
    }
    output = WORKSPACE / "logs" / "qwen3_vl_4b_calibration.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, ensure_ascii=False, indent=2), flush=True)
    print(f"[4b-calibrate] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
