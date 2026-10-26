# UGER: Update-Guided Experience Replay for Continual Graph-Based Recommendation (WWW27')

## Requirements

- Python >= 3.10
- PyTorch >= 2.1
- NumPy, pandas, and SciPy
- scikit-learn and tqdm

```bash
pip install -r requirements.txt
```

## Quickstart: Two-Stage Training

### 1. Pretrain the Recommendation Backbone

```bash
python -u src/main.py \
  --mode pretrain \
  --backbone LightGCN \
  --dataset Amazon-cds \
  --pretrain-epochs 500 \
  --embedding-dim 64 \
  --num-layers 2 \
  --num-negatives 4 \
  --initial-fraction 0.6 \
  --num-snapshots 5 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --batch-size 1024 \
  --seed 2021 \
  --run-name pretrain \
  --gpu 0
```

### 2. Run UGER from the Pretrained Checkpoint

```bash
python -u src/main.py \
  --method UGER \
  --backbone LightGCN \
  --dataset Amazon-cds \
  --pretrained runs/pretrain/pretrained.pt \
  --incremental-epochs 500 \
  --num-negatives 4 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --batch-size 1024 \
  --beta 0.5 \
  --epsilon 0.01 \
  --consistency-weight 1.0 \
  --proposal-size 256 \
  --clean-quota 20 \
  --probe-quota 20 \
  --seed 2021 \
  --run-name uger \
  --gpu 0
```

### 3. Run Experience Replay Baselines

```bash
python -u src/main.py \
  --method Uniform \
  --dataset Amazon-cds \
  --pretrained runs/pretrain/pretrained.pt \
  --seed 2021 \
  --run-name uniform \
  --gpu 0
```

### 4. Run Distillation Baselines


```bash
python -u main.py \
  --model_name Contrastive_LGN \
  --dyn_method finetune-contra-layerwise \
  --dataset Amazon-cds \
  --pretrain_model_path ../runs/pretrain/pretrained.pt \
  --epoch 500 \
  --tepoch 500 \
  --num_neg 4 \
  --lr 0.001 \
  --l2 1e-5 \
  --batch_size 1024 \
  --num_workers 0 \
  --random_seed 2021 \
  --gpu 0
```

## Main Command-Line Options

| Option | Description |
| --- | --- |
| `--buffer-size` | Fixed replay capacity K; defaults to the first incoming-window size. |
| `--beta` | Fusion weight balancing model-aware and structure-aware scores. |
| `--epsilon` | Global parameter perturbation radius. |
| `--consistency-weight` | Weight of the prediction consistency regularizer. |
| `--proposal-size` | Maximum proposal-pool size; default 256. |
| `--clean-quota`, `--probe-quota` | Clean and probe selection quotas; defaults 20 and 20. |
| `--probe-ratio` | Probe quota ratio with the total quota held fixed. |
| `--variant` | Framework ablations and candidate/perturbation controls. |
| `--replay-schedule` | Shared `buffers.npz` for matched replay interactions in candidate controls. |
| `--resume` | Completed-stage checkpoint for resuming training. |


## Analysis and Verification

```bash
PYTHONPATH=src python -m uger.audit --help
PYTHONPATH=src python -m uger.user_groups --help
PYTHONPATH=src python -m uger.reporting --help
PYTHONPATH=src python -m unittest discover -s tests -v
```
