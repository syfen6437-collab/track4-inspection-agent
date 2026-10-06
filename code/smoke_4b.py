from __future__ import annotations

import json
import sys
import time
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))

from track4_agent.data import load_assets
from track4_agent.inference import infer_item
from track4_agent.model import ModelClient


def main() -> None:
    config = json.loads((CODE_DIR / "config" / "qwen3-vl-4b.json").read_text(encoding="utf-8"))
    train, test, lexicon = load_assets(WORKSPACE)
    client = ModelClient(
        WORKSPACE / config["model_dir"],
        max_new_tokens=int(config["max_new_tokens"]),
        global_max_side=int(config["global_max_side"]),
        crop_size=int(config["crop_size"]),
    )
    selected = []
    for category in ("桥梁", "轨道"):
        selected.append(next(item for item in test if item["questionCategory"] == category))
    rows = []
    started = time.time()
    for item in selected:
        item_started = time.time()
        final, raw = infer_item(
            client,
            item,
            lexicon,
            config,
            WORKSPACE / "data" / "raw" / item["image"],
            allow_review=False,
        )
        rows.append(
            {
                "id": item["id"],
                "questionCategory": item["questionCategory"],
                "elapsed_seconds": round(time.time() - item_started, 3),
                "result": final,
                "raw_records": raw,
            }
        )
        print(json.dumps(rows[-1], ensure_ascii=False, indent=2), flush=True)
    report = {
        "model": config["model_id"],
        "samples": rows,
        "elapsed_seconds": round(time.time() - started, 3),
        "gpu": str(getattr(client.torch, "cuda", "")),
    }
    output = WORKSPACE / "logs" / "qwen3_vl_4b_smoke.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] wrote {output}", flush=True)


if __name__ == "__main__":
    main()
