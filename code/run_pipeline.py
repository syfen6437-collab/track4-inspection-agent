from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from track4_agent.data import load_assets, prepare_dataset, read_json
from track4_agent.document import build_design_document
from track4_agent.inference import run_calibration, run_inference
from track4_agent.model import ModelClient, download_model
from track4_agent.validation import build_submission_package, validate_result


DEFAULT_CONFIG = CODE_DIR / "config" / "baseline.json"


def load_config(path: Path) -> dict[str, Any]:
    config = read_json(path)
    config["model_dir"] = str((WORKSPACE / config["model_dir"]).resolve())
    return config


def load_client(config: dict[str, Any]) -> ModelClient:
    # Model weights are downloaded explicitly by the download command. Every
    # calibration or inference client therefore runs in local-only mode.
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["MODELSCOPE_OFFLINE"] = "1"
    return ModelClient(
        Path(config["model_dir"]),
        max_new_tokens=int(config["max_new_tokens"]),
        global_max_side=int(config["global_max_side"]),
        crop_size=int(config["crop_size"]),
    )


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def cmd_prepare(args, config):
    return prepare_dataset(WORKSPACE, Path(args.archive))


def cmd_download(args, config):
    return download_model(config["model_id"], Path(config["model_dir"]))


def cmd_calibrate(args, config):
    train, _, lexicon = load_assets(WORKSPACE)
    client = load_client(config)
    return run_calibration(WORKSPACE, client, config, train, lexicon, args.sample_size)


def cmd_infer(args, config):
    _, test, lexicon = load_assets(WORKSPACE)
    client = load_client(config)
    return run_inference(
        WORKSPACE,
        client,
        config,
        test,
        lexicon,
        resume=args.resume,
        limit=args.limit,
        offset=args.offset,
    )


def cmd_validate(args, config):
    _, test, lexicon = load_assets(WORKSPACE)
    return validate_result(WORKSPACE, test, lexicon)


def cmd_document(args, config):
    return build_design_document(WORKSPACE, team_name=args.team_name)


def cmd_package(args, config):
    _, test, lexicon = load_assets(WORKSPACE)
    validate_result(WORKSPACE, test, lexicon)
    return build_submission_package(WORKSPACE)


def cmd_all(args, config):
    prepare_dataset(WORKSPACE, Path(args.archive))
    download_model(config["model_id"], Path(config["model_dir"]))
    train, test, lexicon = load_assets(WORKSPACE)
    client = load_client(config)
    run_calibration(WORKSPACE, client, config, train, lexicon, args.calibration_sample_size)
    run_inference(WORKSPACE, client, config, test, lexicon, resume=True)
    validate_result(WORKSPACE, test, lexicon)
    build_design_document(WORKSPACE, team_name=args.team_name)
    return build_submission_package(WORKSPACE)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="赛道四本地多模态智能巡检流水线")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="解压数据并生成清单")
    prepare.add_argument("--archive", type=Path, default=WORKSPACE / "赛题四.zip")

    subparsers.add_parser("download", help="从ModelScope下载开源模型")

    calibrate = subparsers.add_parser("calibrate", help="运行本地代理指标校准")
    calibrate.add_argument("--sample-size", type=int, default=120)

    infer = subparsers.add_parser("infer", help="生成测试集预测")
    infer.add_argument("--resume", action="store_true")
    infer.add_argument("--limit", type=int, default=None, help="仅用于GPU冒烟测试")
    infer.add_argument("--offset", type=int, default=0, help="冒烟测试起始序号")

    subparsers.add_parser("validate", help="严格校验result.json")

    document = subparsers.add_parser("document", help="生成智能体设计方案")
    document.add_argument("--team-name", default="")

    package = subparsers.add_parser("package", help="生成tar.gz提交包")
    package.add_argument("--team-name", default="")

    all_parser = subparsers.add_parser("all", help="执行完整流水线")
    all_parser.add_argument("--archive", type=Path, default=WORKSPACE / "赛题四.zip")
    all_parser.add_argument("--calibration-sample-size", type=int, default=120)
    all_parser.add_argument("--team-name", default="")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    set_reproducible(int(config["seed"]))
    handlers = {
        "prepare": cmd_prepare,
        "download": cmd_download,
        "calibrate": cmd_calibrate,
        "infer": cmd_infer,
        "validate": cmd_validate,
        "document": cmd_document,
        "package": cmd_package,
        "all": cmd_all,
    }
    result = handlers[args.command](args, config)
    if result is not None:
        print(json.dumps({"command": args.command, "result": str(result)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
