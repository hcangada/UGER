import argparse
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Update-Guided Experience Replay for continual graph-based recommendation")
    parser.add_argument("--method", choices=["UGER", "Uniform", "Invdeg", "INFER", "Fine-tuning", "Retraining"], default="UGER")
    parser.add_argument("--variant", choices=["full", "w/o-model", "w/o-structure", "w/o-stab", "STAR-Rec", "STAR-Rec-CE", "w/o-CE", "random-probe", "Probe-c", "KL-probe", "w/o-refinement"], default="full")
    parser.add_argument("--mode", choices=["pretrain", "continual"], default="continual")
    parser.add_argument("--backbone", "--model_name", default="LightGCN")
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--data-root", "--path", type=Path, default=ROOT / "data")
    parser.add_argument("--dataset", default="Amazon-cds")
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--initial-fraction", "--train_ratio", type=float, default=0.6)
    parser.add_argument("--num-snapshots", "--n_snapshots", type=int, default=5)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--split-mode", choices=["count", "time"], default="count")
    parser.add_argument("--embedding-dim", "--emb_size", type=int, default=64)
    parser.add_argument("--num-layers", "--n_layers", type=int, default=2)
    parser.add_argument("--keep-prob", type=float, default=1.0)
    parser.add_argument("--message-dropout", type=float, default=0.0)
    parser.add_argument("--mgccf-neighbors", type=int, nargs="+")
    parser.add_argument("--similarity-degree", type=int, default=10)
    parser.add_argument("--pretrain-epochs", "--epoch", type=int, default=500)
    parser.add_argument("--incremental-epochs", "--tepoch", type=int, default=500)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=1024)
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", "--l2", type=float, default=1e-5)
    parser.add_argument("--num-negatives", "--num_neg", type=int, default=4)
    parser.add_argument("--buffer-size", "--K", type=int)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--consistency-weight", "--lambda_", type=float, default=1.0)
    parser.add_argument("--epsilon", "--eps", type=float, default=0.01)
    parser.add_argument("--proposal-size", "--n_p", type=int, default=256)
    parser.add_argument("--clean-quota", "--k_c", type=int, default=20)
    parser.add_argument("--probe-quota", "--k_u", type=int, default=20)
    parser.add_argument("--probe-ratio", "--sigma", type=float)
    parser.add_argument("--candidate-policy", choices=["enhanced", "clean", "probe"], default="enhanced")
    parser.add_argument("--perturbation-init", choices=["incoming", "random", "kl"], default="incoming")
    parser.add_argument("--candidate-probe", choices=["incoming", "random", "initial"], default="incoming")
    parser.add_argument("--random-scale", type=float, default=0.01)
    parser.add_argument("--normalization-floor", type=float, default=1e-12)
    parser.add_argument("--fallback-scale", type=float, default=0.01)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--selection-batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--evaluation-interval", type=int, default=1)
    parser.add_argument("--evaluation-batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--seed", "--random_seed", type=int, default=2021)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--run-name")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--replay-schedule", type=Path)
    parser.add_argument("--save-every-stage", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.mgccf_neighbors is None:
        args.mgccf_neighbors = [15 if layer == 0 else 10 for layer in range(args.num_layers)]
    for name in ("initial_fraction", "validation_fraction", "keep_prob", "message_dropout", "learning_rate", "weight_decay", "beta", "consistency_weight", "epsilon", "normalization_floor", "fallback_scale", "random_scale"):
        if not math.isfinite(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be finite")
    variants = {
        "w/o-model": {"beta": 0.0},
        "w/o-structure": {"beta": 1.0},
        "w/o-stab": {"consistency_weight": 0.0},
        "STAR-Rec": {"candidate_policy": "clean", "perturbation_init": "random", "candidate_probe": "incoming", "no_refinement": False},
        "STAR-Rec-CE": {"candidate_policy": "enhanced", "perturbation_init": "random", "candidate_probe": "incoming", "no_refinement": False},
        "w/o-CE": {"candidate_policy": "clean", "perturbation_init": "incoming", "candidate_probe": "incoming", "no_refinement": False},
        "random-probe": {"candidate_policy": "enhanced", "candidate_probe": "random", "perturbation_init": "incoming", "no_refinement": False},
        "Probe-c": {"candidate_policy": "probe", "perturbation_init": "incoming", "candidate_probe": "incoming", "no_refinement": False},
        "KL-probe": {"candidate_policy": "enhanced", "perturbation_init": "kl", "candidate_probe": "initial", "no_refinement": False},
        "w/o-refinement": {"candidate_policy": "enhanced", "perturbation_init": "incoming", "candidate_probe": "incoming", "no_refinement": True},
    }
    if args.method != "UGER" and args.variant != "full":
        parser.error("Variants apply only to UGER")
    if args.replay_schedule and (args.method != "UGER" or args.variant in {"w/o-model", "w/o-structure"}):
        parser.error("Shared replay schedules require UGER with the fusion sampling strategy")
    for key, value in variants.get(args.variant, {}).items():
        setattr(args, key, value)
    if args.probe_ratio is not None:
        total = args.clean_quota + args.probe_quota
        if not 0 <= args.probe_ratio <= 1 or total == 0:
            parser.error("probe-ratio requires a positive total quota and a value in [0, 1]")
        probe = args.probe_ratio * total
        if abs(probe - round(probe)) > 1e-9:
            parser.error("probe-ratio must allocate an integer number of candidate slots")
        args.probe_quota = int(round(probe))
        args.clean_quota = total - args.probe_quota
    for name in ["num_snapshots", "embedding_dim", "pretrain_epochs", "incremental_epochs", "batch_size", "num_negatives", "proposal_size", "selection_batch_size", "evaluation_interval", "evaluation_batch_size", "num_threads"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.num_layers < 0 or args.patience < 0 or min(args.clean_quota, args.probe_quota) < 0:
        parser.error("Layer count, patience, and candidate quotas must be nonnegative")
    if args.buffer_size is not None and args.buffer_size < 0:
        parser.error("buffer-size must be nonnegative")
    if args.mode == "continual" and args.method in {"UGER", "Uniform", "Invdeg", "INFER"} and args.buffer_size != 0 and args.batch_size < 2:
        parser.error("Incoming and replay batches require --batch-size of at least two")
    if not 0 < args.initial_fraction < 1 or not 0 < args.validation_fraction < 1:
        parser.error("Initial and validation fractions must lie strictly between zero and one")
    if not 0 <= args.beta <= 1:
        parser.error("beta must lie in [0, 1]")
    if args.learning_rate <= 0 or args.epsilon <= 0 or args.normalization_floor <= 0:
        parser.error("Learning rate, epsilon, and normalization floor must be positive")
    if args.weight_decay < 0 or args.consistency_weight < 0 or not 0 < args.fallback_scale <= 1:
        parser.error("Loss weights must be nonnegative; fallback scale must lie in (0, 1]")
    if not 0 < args.random_scale <= 1:
        parser.error("random-scale must lie in (0, 1]")
    if args.similarity_degree < 1 or any(value < 1 for value in args.mgccf_neighbors):
        parser.error("MGCCF neighbor counts must be positive")
    if args.backbone.lower().replace("-", "").replace("_", "") in {"mgccf", "multigccf"} and len(args.mgccf_neighbors) != args.num_layers:
        parser.error("MGCCF neighbor counts must match the number of layers")
    if not 0 < args.keep_prob <= 1 or not 0 <= args.message_dropout < 1:
        parser.error("keep-prob must lie in (0, 1]; message-dropout must lie in [0, 1)")
    if args.clean_quota + args.probe_quota + 1 > args.proposal_size:
        parser.error("clean-quota + probe-quota + 1 cannot exceed proposal-size")
    if any(k <= 0 for k in args.top_k):
        parser.error("All ranking cutoffs must be positive")
    if args.pretrained and args.resume:
        parser.error("Choose either --pretrained or --resume")
    args.top_k = sorted(set(args.top_k))
    if args.data_file is None:
        args.data_file = args.data_root / args.dataset / "interactions" / "freq.csv"
    return args
