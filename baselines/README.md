# Comparison Baselines

Run from this directory after installing the project requirements. Input files use the same format and ID mapping as the main implementation. The distillation and PISA routes use LightGCN.

Use the stage-zero LightGCN checkpoint from the main implementation:

```bash
python main.py --model_name Contrastive_LGN --dyn_method finetune-contra-layerwise --dataset Amazon-cds --pretrain_model_path ../runs/pretrain/pretrained.pt
```

| Method | Model | Strategy |
| --- | --- | --- |
| Fine-tuning | LGN | finetune |
| Retraining | LGN | fulltrain |
| Uniform | LGN | finetune-uniform-curr |
| Invdeg | LGN | finetune-invdeg-curr |
| INFER | LGN | finetune-infer-curr |
| LWC-KD | Contrastive_LGN | finetune-contra-layerwise |
| LWC-KD-PIW | Contrastive_LGN | finetune-contra-piw |
| PISA | PISA_LGN | finetune-stability-plasticity-userneigh |

Set `--model_name` and `--dyn_method` using the table. Defaults use a 60% initial block and five equal-count incoming windows. Each future window is split equally into validation and test subsets, with the same pre-window history mask and full-catalogue ranking. Replay capacity defaults to the first incoming-window size; `--buffer-size` overrides it.

The shared checkpoint must match the dataset, seed, dimensions, layers, and temporal split. `--epoch`, `--tepoch`, and `--batch_size` control training. Use `--gpu -1 --num_workers 0` for CPU execution. Run `python main.py --help` for other options.
