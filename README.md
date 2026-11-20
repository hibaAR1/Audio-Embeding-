# CrossGated

Code and results for the article *CrossGated: Adaptive Fusion of Pretrained Convolutional and Transformer Representations for Environmental Sound Classification* (H. Arbel, A. El Oirrak).

CrossGated fuses the frozen embeddings of PANNs CNN14 (2048-d) and AST (768-d) with cross gates and a Fisher loss. Only a small head (537,714 trainable parameters, about 0.54 M) is trained. It is evaluated on ESC-50 and UrbanSound8K with their predefined folds and 3 seeds.

## Installation

Python 3.11 or newer.

```
python -m venv .venv
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Data

The datasets are public and are not included:

- ESC-50: https://github.com/karolpiczak/ESC-50
- UrbanSound8K: https://zenodo.org/record/1203745

## Reproducing the results

**1. Extract the frozen embeddings** (the PANNs checkpoint and the AST weights are downloaded automatically). Features are written to `features_v2/<dataset>/`.

```
python extract_features.py esc50 --data path/to/ESC-50
python extract_features.py urban --data path/to/UrbanSound8K
```

**2. Train and evaluate the seven variants** (SVM, PANNs only, AST only, concatenation, uncrossed gates, no Fisher loss, CrossGated) on all predefined folds and seeds 0, 1, 2. Results are written to `results/<dataset>_results.csv`. The script resumes where it stopped if interrupted.

```
python crossgated.py esc50
python crossgated.py urban
python crossgated.py summary results/urban_results.csv
```

The summary prints the mean accuracy and the paired Wilcoxon tests over folds. Use `--restart` only to start again from zero (the previous files are renamed, not deleted).

**3. Complementarity, gates and per-class figure.** Save the predictions of the three main models, then run the analyses:

```
python crossgated.py esc50 --models panns_only ast_only crossgated --save-preds --out results_preds
python crossgated.py urban --models panns_only ast_only crossgated --save-preds --out results_preds
python crossgated.py esc50 --models crossgated --seeds 0 --out results_gates
python crossgated.py urban --models crossgated --seeds 0 --out results_gates
python analyse_predictions.py esc50 --esc50-meta ESC-50/meta/esc50.csv
python analyse_predictions.py urban
python analyse_gates.py esc50 --esc50-meta ESC-50/meta/esc50.csv
python analyse_gates.py urban
python fig_classwise.py
```

The gate analysis uses seed 0 only. `fig_classwise.py` draws Figure 2 of the article.

**4. Computational cost** (parameters and CPU inference time per 5 s clip). Close other programs first, because the times depend on the load of the machine.

```
python measure_cost.py
```

## Files

| File | Role |
|---|---|
| `extract_features.py` | Frozen PANNs CNN14 and AST embeddings |
| `crossgated.py` | Model, variants, training, evaluation, statistics |
| `analyse_predictions.py` | Complementarity of the two encoders, gain per class |
| `analyse_gates.py` | Gate values per class |
| `fig_classwise.py` | Per-class accuracy figure (Figure 2) |
| `measure_cost.py` | Parameters and inference time |
| `results/`, `results_preds/`, `results_gates/`, `results_cost/` | Outputs |

## Notes

- The AST embedding is the mean of all last-layer tokens (`ast.npy`). AST pads each input to 1024 frames (about 10 s), so this mean includes the padded positions. `ast_real.npy` keeps only the patches that cover real audio; select it with `--ast-file ast_real.npy`.
- Only the head is trained. Inference still needs both encoders.
- Results can differ slightly between machines (CPU/GPU, library versions).

## License

MIT, see `LICENSE`.
