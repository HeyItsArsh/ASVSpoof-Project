"""One command per reproducible project step: python -m secure_audio --help."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .common import read_config, save_json


def main() -> None:
    """Dispatch explicit steps without implicitly training or touching eval audio."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-data", help="Validate official layout and every protocol trial")
    check.add_argument("--content-hash", action="store_true")
    train = sub.add_parser("train", help="Fit on train, select on dev; never score eval")
    train.add_argument("--model", choices=["svm", "boosting", "mobile", "all"], default="all")
    train.add_argument("--seed", type=int)
    train.add_argument("--resume", action="store_true", help="Skip completed, matching runs; restart incomplete runs from seed")
    freeze = sub.add_parser("freeze", help="Lock completed runs before eval")
    freeze.add_argument("--root", type=Path)
    evaluate = sub.add_parser("evaluate", help="Score frozen models on official eval")
    evaluate.add_argument("--run", type=Path)
    aggregate = sub.add_parser("aggregate", help="Summarize three seeds and attack-level results")
    aggregate.add_argument("--root", type=Path)
    sub.add_parser("ablations", help="Write four one-factor deep model ablation configs")
    sanity = sub.add_parser("sanity", help="Run diagnostic controls, never benchmark results")
    sanity.add_argument("--real-data", action="store_true")
    sanity.add_argument("--steps", type=int, default=200)
    sanity.add_argument("--only", choices=["all", "overfit", "shuffle", "shuffle-deep", "edges"], default="all")
    infer = sub.add_parser("infer", help="Classify a research clip with explicit rejection handling")
    infer.add_argument("--run", type=Path, required=True)
    infer.add_argument("--audio", type=Path, required=True)
    external = sub.add_parser("external", help="Score a separate local TTS spoof-only probe")
    external.add_argument("--run", type=Path, required=True)
    external.add_argument("--folder", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()
    cfg = read_config(args.config)
    root = Path(cfg["output_root"])
    if args.command == "check-data":
        from .data import audit
        data_root = Path(cfg["data_root"])
        if not data_root.exists():
            free = shutil.disk_usage(Path.cwd()).free / 1024**3
            parser.exit(2, f"Missing {data_root}. Download LA.zip (not PA.zip) from https://datashare.ed.ac.uk/handle/10283/3336\n"
                        f"Extract so {data_root}/ASVspoof2019_LA_train/flac and {data_root}/ASVspoof2019_LA_cm_protocols exist.\n"
                        f"Automatic: python -m secure_audio.download --extract\nFree space: {free:.1f} GiB; reserve 30+ GiB including archive/cache.\n")
        try:
            result = audit(data_root, content_hash=args.content_hash)
        except FileNotFoundError as exc:
            parser.exit(2, f"Missing official file: {exc.filename or exc}\n"
                        f"Extract LA.zip from https://datashare.ed.ac.uk/handle/10283/3336 into data/, keeping LA/ intact.\n"
                        f"Expected protocols under {data_root}/ASVspoof2019_LA_cm_protocols and audio under\n"
                        f"{data_root}/ASVspoof2019_LA_{{train,dev,eval}}/flac.\n"
                        "Automatic: python -m secure_audio.download --extract\n")
        save_json(data_root / "audit.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "train":
        from .train import fit_baseline, fit_deep, run_directory
        if (root / "frozen.json").exists():
            raise ValueError("Experiment is frozen; use a fresh output_root for new experiments")
        for name in (["svm", "boosting", "mobile"] if args.model == "all" else [args.model]):
            for seed in ([args.seed] if args.seed is not None else cfg["seeds"]):
                directory = run_directory(cfg, name, seed)
                if args.resume and (directory / "run.json").exists():
                    old = json.loads((directory / "run.json").read_text(encoding="utf-8"))
                    checkpoint = directory / ("model.pt" if name == "mobile" else "model.joblib")
                    required = [checkpoint, directory / "dev/metrics.json", directory / "dev/scores.csv"]
                    if (old["config"] != cfg or old["seed"] != seed or old["model"] != name
                            or not all(p.exists() for p in required)):
                        raise ValueError(f"Cannot resume mismatching or incomplete completed run: {directory}")
                    print(f"Already complete: {directory}", flush=True)
                    continue
                out = fit_deep(cfg, seed) if name == "mobile" else fit_baseline(cfg, name, seed)
                print(f"Completed: {out}", flush=True)
    elif args.command == "freeze":
        from .experiments import freeze_experiments
        freeze_experiments(args.root or root, cfg["seeds"])
        print("Experiment frozen. Evaluate with the same config.")
    elif args.command == "evaluate":
        from .experiments import verify_frozen
        from .train import evaluate_run
        runs = [args.run] if args.run else sorted(root.glob("runs/*/seed_*"))
        if not runs:
            raise ValueError("No runs to evaluate")
        for run in runs:
            verify_frozen(run)
            result = evaluate_run(run)
            print(f"{run}: eval EER={result['eer']:.6f}", flush=True)
    elif args.command == "aggregate":
        from .experiments import aggregate as aggregate_runs
        aggregate_runs(args.root or root, cfg["seeds"])
        print("Wrote aggregate.json (fractions, not percentages)")
    elif args.command == "ablations":
        from .experiments import make_ablations
        print("\n".join(str(p) for p in make_ablations(args.config)))
    elif args.command == "sanity":
        from .sanity import edge_cases, overfit, shuffled_deep, shuffled_labels
        results = {}
        if args.only in {"all", "overfit"}:
            results["overfit"] = overfit(cfg, real_data=args.real_data, steps=args.steps)
        if args.only in {"all", "shuffle"}:
            results["shuffled_labels"] = shuffled_labels(cfg, real_data=args.real_data)
        if args.only in {"all", "shuffle-deep"}:
            results["shuffled_deep"] = shuffled_deep(cfg, real_data=args.real_data)
        if args.only in {"all", "edges"}:
            results["edge_cases"] = edge_cases(cfg)
        output = root / ("sanity_real.json" if args.real_data else "sanity_synthetic.json")
        save_json(output, results)
        print(json.dumps(results, indent=2))
        if not all(value["passed"] for value in results.values()):
            raise SystemExit("A sanity check failed; inspect saved diagnostics before running experiments")
    elif args.command == "infer":
        from .inference import Detector
        print(json.dumps(Detector(args.run).score(args.audio), indent=2))
    elif args.command == "external":
        from .inference import external_probe
        result = external_probe(args.run, args.folder)
        save_json(args.run / "external" / "probe.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
