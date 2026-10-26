import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel


VARIANT_NAMES = {
    "full": "UGER",
    "w/o-model": "UGER (w/o Model)",
    "w/o-structure": "UGER (w/o Structure)",
    "w/o-stab": "UGER (w/o STAB)",
    "STAR-Rec": "STAR-Rec",
    "STAR-Rec-CE": "STAR-Rec + CE",
    "w/o-CE": "UGER (w/o CE)",
    "random-probe": "Random-Probe",
    "Probe-c": "Probe-c",
    "KL-probe": "KL-probe",
    "w/o-refinement": "UGER (w/o Ref.)",
}


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("P-values must be a finite one-dimensional array in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = np.empty_like(values)
    bound = 0.0
    for rank, index in enumerate(order):
        bound = max(bound, min(1.0, (len(values) - rank) * values[index]))
        adjusted[index] = bound
    return adjusted.tolist()


def load_run(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    stages = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if not stages or any("validation" not in stage for stage in stages):
        raise ValueError("Reporting requires completed runs with separate next-window validation and test metrics")
    windows = tuple(stage["evaluated_on_stage"] for stage in stages)
    if len(set(windows)) != len(windows):
        raise ValueError("A prediction window is repeated within a run")
    metrics = sorted(stages[0]["metrics"])
    if any(sorted(stage["metrics"]) != metrics or not stage["evaluated_users"] for stage in stages):
        raise ValueError("Each evaluated window must contain the same metrics and eligible targets")
    values = {metric: float(np.mean([stage["metrics"][metric] for stage in stages])) for metric in metrics}
    if not np.isfinite(list(values.values())).all():
        raise ValueError("Run-level metrics must be finite")
    name = config["method"]
    if name == "UGER":
        name = VARIANT_NAMES[config.get("variant", "full")]
    protocol = (config["data_fingerprint"], config["initial_fraction"], config["num_snapshots"],
                config["split_mode"], config.get("validation_fraction"), config.get("buffer_size"),
                tuple(metrics), windows)
    return {"dataset": config["dataset"], "backbone": config["backbone"], "seed": int(config["seed"]),
            "method": name, "protocol": protocol, "values": values,
            "mean_stage_seconds": summary.get("mean_incremental_stage_seconds")}


def aggregate_runs(directories, reference="UGER", alpha=0.05, required_seeds=5):
    if not 0 < alpha < 1 or required_seeds < 2:
        raise ValueError("Use alpha in (0, 1) and at least two seeds")
    grouped = defaultdict(lambda: defaultdict(dict))
    protocols = {}
    for directory in directories:
        run = load_run(directory)
        key = (run["dataset"], run["backbone"])
        if key in protocols and protocols[key] != run["protocol"]:
            raise ValueError("Compared runs must use identical data, splits, and prediction windows")
        protocols[key] = run["protocol"]
        seeds = grouped[key][run["method"]]
        if run["seed"] in seeds:
            raise ValueError("Each method must have one run per seed")
        seeds[run["seed"]] = run
    output = []
    for (dataset, backbone), methods in sorted(grouped.items()):
        if reference not in methods:
            raise ValueError(f"Missing reference method: {reference}")
        seed_ids = sorted(methods[reference])
        if len(seed_ids) != required_seeds or any(sorted(runs) != seed_ids for runs in methods.values()):
            raise ValueError("All compared methods must have the required identical seed set")
        metrics = sorted(next(iter(methods[reference].values()))["values"])
        rows = {}
        for metric in metrics:
            samples = {name: np.asarray([runs[seed]["values"][metric] for seed in seed_ids]) for name, runs in methods.items()}
            estimates = {name: {"mean_percent": float(values.mean() * 100),
                                "sd_percent": float(values.std(ddof=1) * 100),
                                "run_means_percent": (values * 100).tolist()}
                         for name, values in samples.items()}
            comparators = sorted(name for name in methods if name != reference)
            raw = []
            for name in comparators:
                difference = samples[reference] - samples[name]
                pvalue = None if difference.var(ddof=1) == 0 else float(ttest_rel(samples[reference], samples[name]).pvalue)
                raw.append(pvalue if pvalue is not None and np.isfinite(pvalue) else None)
            adjusted = holm_adjust([1.0 if value is None else value for value in raw])
            tests = {name: {"p_raw": raw[index],
                            "p_holm": adjusted[index] if raw[index] is not None else None,
                            "reference_mean_higher": bool(samples[reference].mean() > samples[name].mean())}
                     for index, name in enumerate(comparators)}
            significant = bool(tests) and all(test["reference_mean_higher"] and test["p_holm"] is not None and test["p_holm"] <= alpha for test in tests.values())
            rows[metric] = {"estimates": estimates, "comparisons": tests,
                            "significant_against_all_provided_comparators": significant}
        output.append({"dataset": dataset, "backbone": backbone, "seeds": seed_ids,
                       "reference": reference, "test": "two-sided paired t-test", "correction": "Holm",
                       "alpha": alpha, "metrics": rows})
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aggregate measured run-level metrics with paired Holm-corrected tests")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--reference", default="UGER")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--required-seeds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = aggregate_runs(args.runs, args.reference, args.alpha, args.required_seeds)
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
