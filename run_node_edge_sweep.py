from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUPPORTED_DATASETS = [
    "cora",
    "citeseer",
    "pubmed",
    "coauthor_cora",
    "coauthor_dblp",
    "20newsW100",
    "ModelNet40",
    "zoo",
    "NTU2012",
    "Mushroom",
    "yelp",
    "walmart-trips-100",
    "house-committees-100",
    "actor",
    "amazon",
    "pokec",
    "twitch",
    "german",
    "bail",
    "credit",
    "amazon_review",
    "magpm",
    "trivago",
    "ogbn_mag",
]

SUPPORTED_METHODS = [
    "HGNN",
    "HNHN",
    "MLP",
    "UniGIN",
    "UniGCNII",
    "AllSetformer",
    "AllDeepSets",
]

SUPPORTED_TASKS = ["node_cls", "edge_pred"]
SUPPORTED_PIPELINES = ["subgraph", "baseline"]


def parse_csv(value: str, allowed: list[str], field: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in items if item not in allowed]
    if unknown:
        raise ValueError(f"Unknown {field}: {unknown}. Allowed: {allowed}")
    return items


def add_common_args(cmd: list[str], args, task: str, pipeline: str) -> list[str]:
    cmd.extend(
        [
            "--device",
            args.device,
            "--num_seeds",
            str(args.num_seeds),
            "--display_step",
            str(args.display_step),
        ]
    )
    for flag_name, value in (
        ("--epochs", args.epochs),
        ("--dropout", args.dropout),
        ("--lr", args.lr),
        ("--wd", args.wd),
    ):
        if value is not None:
            cmd.extend([flag_name, str(value)])

    if task == "edge_pred":
        cmd.extend(
            [
                "--edge_split_mode",
                args.edge_split_mode,
                "--edge_batch_size",
                str(args.edge_batch_size),
                "--aggr_mode",
                args.aggr_mode,
                "--ns_method",
                args.ns_method,
            ]
        )

    if pipeline == "subgraph":
        cmd.extend(
            [
                "--lr",
                str(args.subgraph_lr),
                "--dropout",
                str(args.subgraph_dropout),
                "--subgraph_max_hyperedges",
                str(args.subgraph_max_hyperedges),
                "--subgraph_context_hops",
                str(args.subgraph_context_hops),
                "--subgraph_batch_size",
                str(args.subgraph_batch_size),
                "--subgraph_cache",
                str(args.subgraph_cache).lower(),
            ]
        )

    if args.extra_args:
        cmd.extend(shlex.split(args.extra_args))
    if task == "node_cls" and args.extra_node_args:
        cmd.extend(shlex.split(args.extra_node_args))
    if task == "edge_pred" and args.extra_edge_args:
        cmd.extend(shlex.split(args.extra_edge_args))
    if pipeline == "subgraph" and args.extra_subgraph_args:
        cmd.extend(shlex.split(args.extra_subgraph_args))
    if pipeline == "baseline" and args.extra_baseline_args:
        cmd.extend(shlex.split(args.extra_baseline_args))

    return cmd


def build_command(args, dataset: str, method: str, task: str, pipeline: str) -> list[str]:
    cmd = [
        args.python,
        "main.py",
        "--pipeline",
        pipeline,
        "--dname",
        dataset,
        "--task_type",
        task,
        "--method",
        method,
    ]
    return add_common_args(cmd, args, task, pipeline)


def log_path_for(log_dir: Path, dataset: str, method: str, task: str, pipeline: str) -> Path:
    safe_dataset = dataset.replace("/", "_")
    return log_dir / task / pipeline / method / f"{safe_dataset}.log"


def is_completed(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return False
    return "---------------------------------[Final]--------------------------------------" in text


def run_command(cmd: list[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    header = "\n".join(
        [
            "=" * 100,
            f"START {datetime.now().isoformat(timespec='seconds')}",
            f"CWD   {cwd}",
            "CMD   " + shlex.join(cmd),
            "=" * 100,
            "",
        ]
    )

    print(header, flush=True)
    with log_path.open("a", buffering=1) as log_file:
        log_file.write(header)
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
        footer = "\n".join(
            [
                "",
                "=" * 100,
                f"END   {datetime.now().isoformat(timespec='seconds')}",
                f"CODE  {return_code}",
                "=" * 100,
                "",
            ]
        )
        print(footer, flush=True)
        log_file.write(footer)
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run node_cls and edge_pred sweeps and keep raw console logs.")
    parser.add_argument("--datasets", default="all", help="Comma list or all.")
    parser.add_argument("--methods", default="all", help="Comma list or all.")
    parser.add_argument("--tasks", default="node_cls,edge_pred", help="Comma list from node_cls,edge_pred.")
    parser.add_argument("--pipelines", default="subgraph", help="Comma list from subgraph,baseline.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--epochs", type=int, default=None, help="Default None means use lib_yamls config.")
    parser.add_argument("--num-seeds", "--num_seeds", dest="num_seeds", type=int, default=5)
    parser.add_argument("--display-step", "--display_step", dest="display_step", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=None, help="Default None means use lib_yamls config.")
    parser.add_argument("--lr", type=float, default=None, help="Default None means use lib_yamls config.")
    parser.add_argument("--wd", type=float, default=None, help="Default None means use lib_yamls config.")
    parser.add_argument("--edge-split-mode", "--edge_split_mode", dest="edge_split_mode", default="trand")
    parser.add_argument("--edge-batch-size", "--edge_batch_size", dest="edge_batch_size", type=int, default=512)
    parser.add_argument("--aggr-mode", "--aggr_mode", dest="aggr_mode", default="maxmin", choices=["max", "mean", "maxmin"])
    parser.add_argument("--ns-method", "--ns_method", dest="ns_method", default="mixed", choices=["mns", "sns", "cns", "mixed"])
    parser.add_argument("--subgraph-lr", "--subgraph_lr", dest="subgraph_lr", type=float, default=0.0001)
    parser.add_argument("--subgraph-dropout", "--subgraph_dropout", dest="subgraph_dropout", type=float, default=0.6)
    parser.add_argument("--subgraph-max-hyperedges", "--subgraph_max_hyperedges", dest="subgraph_max_hyperedges", type=int, default=8)
    parser.add_argument("--subgraph-context-hops", "--subgraph_context_hops", dest="subgraph_context_hops", type=int, default=1)
    parser.add_argument("--subgraph-batch-size", "--subgraph_batch_size", dest="subgraph_batch_size", type=int, default=256)
    parser.add_argument("--subgraph-cache", "--subgraph_cache", dest="subgraph_cache", default="true", choices=["true", "false"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--start-at", type=int, default=0, help="Zero-based command index to start from.")
    parser.add_argument("--extra-args", default="", help="Extra args appended to every main.py call.")
    parser.add_argument("--extra-node-args", default="", help="Extra args appended only for node_cls.")
    parser.add_argument("--extra-edge-args", default="", help="Extra args appended only for edge_pred.")
    parser.add_argument("--extra-subgraph-args", default="", help="Extra args appended only for subgraph pipeline.")
    parser.add_argument("--extra-baseline-args", default="", help="Extra args appended only for baseline pipeline.")
    args = parser.parse_args()

    datasets = parse_csv(args.datasets, SUPPORTED_DATASETS, "datasets")
    methods = parse_csv(args.methods, SUPPORTED_METHODS, "methods")
    tasks = parse_csv(args.tasks, SUPPORTED_TASKS, "tasks")
    pipelines = parse_csv(args.pipelines, SUPPORTED_PIPELINES, "pipelines")

    repo_dir = Path(__file__).resolve().parent
    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_dir = repo_dir / "runs" / f"node_edge_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    jobs = []
    for task in tasks:
        for pipeline in pipelines:
            for method in methods:
                for dataset in datasets:
                    cmd = build_command(args, dataset, method, task, pipeline)
                    log_path = log_path_for(log_dir, dataset, method, task, pipeline)
                    jobs.append((cmd, log_path, task, pipeline, method, dataset))

    selected_jobs = jobs[args.start_at :]
    if args.max_runs > 0:
        selected_jobs = selected_jobs[: args.max_runs]

    print(f"Planned jobs: {len(jobs)} total, {len(selected_jobs)} selected")
    print(f"Log dir: {log_dir}")

    failures = []
    for offset, (cmd, log_path, task, pipeline, method, dataset) in enumerate(selected_jobs, start=args.start_at):
        label = f"[{offset + 1}/{len(jobs)}] {task} | {pipeline} | {method} | {dataset}"
        if args.skip_existing and is_completed(log_path):
            print(f"SKIP completed {label}: {log_path}")
            continue
        print(f"RUN {label}")
        print(shlex.join(cmd))
        if args.dry_run:
            continue
        return_code = run_command(cmd, log_path, repo_dir)
        if return_code != 0:
            failures.append((return_code, label, log_path))
            print(f"FAILED code={return_code} {label}: {log_path}", flush=True)
            if args.stop_on_error:
                break

    if failures:
        print("Failures:")
        for return_code, label, log_path in failures:
            print(f"  code={return_code} {label} {log_path}")
        return 1

    print("All selected jobs finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
