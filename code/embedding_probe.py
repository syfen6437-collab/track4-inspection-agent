from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from track4_agent.data import load_assets
from track4_agent.model import ModelClient, resize_global, scene_key


def main() -> None:
    config = json.loads((CODE / "config" / "qwen3-vl-4b-highres.json").read_text(encoding="utf-8"))
    train, _, lexicon = load_assets(ROOT)
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rng = random.Random(int(config["seed"]))

    # Hold out a balanced sample; prototypes never include the query row.
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in train:
        item = {**row, "questionCategory": row["questionCategory"], "filename": row["filename"]}
        buckets[(row["questionCategory"], scene_key(item))].append(row)
    sample: list[dict] = []
    for rows in buckets.values():
        if rows:
            sample.append(rng.choice(rows))
    rng.shuffle(sample)
    sample = sample[:sample_size]

    client = ModelClient(
        ROOT / config["model_dir"],
        max_new_tokens=8,
        global_max_side=320,
        crop_size=160,
    )
    features: dict[str, object] = {}
    def get_feature(row: dict) -> object:
        key = row["image"]
        if key not in features:
            with Image.open(ROOT / "data" / "raw" / row["image"]) as image:
                features[key] = client.embed_image(resize_global(image, 320))
        return features[key]

    # A compact prototype bank is enough to test whether the vision space has
    # useful structure without extracting all 3,402 images.
    prototypes: dict[tuple[str, str, str], list] = defaultdict(list)
    selected_prototypes: dict[tuple[str, str, str], dict] = {}
    for row in train:
        if row in sample:
            continue
        item = {**row, "questionCategory": row["questionCategory"], "filename": row["filename"]}
        key = (row["questionCategory"], scene_key(item), row["defectType"])
        # One representative per label is enough for a speed probe. A full
        # prototype bank is only worthwhile after this test beats generation.
        if key not in selected_prototypes:
            selected_prototypes[key] = row
    for key, row in selected_prototypes.items():
        prototypes[key].append(get_feature(row))

    correct = 0
    rows_out = []
    for index, row in enumerate(sample, start=1):
        item = {**row, "questionCategory": row["questionCategory"], "filename": row["filename"]}
        query = get_feature(row)
        qscene = scene_key(item)
        candidates = []
        for (category, scene, label), values in prototypes.items():
            if category != row["questionCategory"] or (scene != qscene and qscene != "generic"):
                continue
            centroid = sum(values) / len(values)
            centroid = centroid / max(float(centroid.norm()), 1e-8)
            score = float(query @ centroid)
            candidates.append((score, label))
        candidates.sort(reverse=True)
        prediction = candidates[0][1] if candidates else ""
        correct += int(prediction == row["defectType"])
        rows_out.append({"truth": row["defectType"], "prediction": prediction, "scene": qscene, "top": candidates[:5]})
        print(f"[embed] {index}/{len(sample)} {row['questionCategory']} {row['filename']} -> {prediction}", flush=True)

    report = {
        "model": config["model_id"],
        "sample_size": len(sample),
        "exact_accuracy": correct / len(sample) if sample else 0.0,
        "prototype_count": sum(len(v) for v in prototypes.values()),
        "rows": rows_out,
    }
    output = ROOT / "logs" / "embedding_probe.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
