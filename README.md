# Traffic Trace Classification Project

This project contains a small experiment pipeline for traffic trace classification using burst-based tokenization, TF-IDF baselines, and a transformer classifier.

## Repository contents

- `1_data_and_tokenization.ipynb`
  - Loads the dataset from the `.npz` archive.
  - Defines trace loading utilities that support both `.npz` and raw directory formats.
  - Implements burst-based tokenization:
    - `burst_tokenize` produces coarse tokens like `OUT_S` and `IN_M`.
    - `proposal_burst_tokenize` produces proposal-style tokens with length buckets like `OUT_LEN_017_018`.
  - Builds vocabularies and prints sample token statistics.

- `2_baselines.ipynb`
  - Loads the same dataset and trace data.
  - Converts traces to document strings using burst tokenization.
  - Evaluates a classic TF-IDF + logistic regression baseline.
  - Trains a BPE tokenizer on proposal-style trace tokens.
  - Evaluates a BPE + TF-IDF baseline for comparison.

- `3_transformer_training.ipynb`
  - Loads the dataset and prepares proposal-style burst tokens.
  - Trains a BPE tokenizer and uses it to tokenize traces.
  - Creates a PyTorch transformer classifier for multi-class prediction.
  - Trains with a train/validation split and saves the best model, tokenizer, and training summary.

- `tor_100w_2500tr.npz`
- `tor_200w_2500tr.npz`
- `tor_500w_2500tr.npz`
- `tor_900w_2500tr.npz`
  - Dataset files containing traffic traces.
  - The notebooks currently use `tor_100w_2500tr.npz` by default.
  - The larger `.npz` files are available for scaling experiments.
  - Dataset source: https://distrinet.cs.kuleuven.be/software/tor-wf-dl/

## How to use this project

### Environment setup

Run the notebooks in a Python environment with the required libraries.

The notebooks install dependencies at runtime, but a clean environment should include at least:

- Python 3.10+ (or compatible)
- `numpy`
- `scikit-learn`
- `tokenizers`
- `torch` (for transformer training)

If running locally, install the libraries manually with:

```bash
python -m pip install numpy scikit-learn tokenizers torch
```

### Notebook order

1. `1_data_and_tokenization.ipynb`
   - Verify dataset loading and inspect tokenization.
   - Run this notebook first to confirm the data format and token vocabulary.

2. `2_baselines.ipynb`
   - Run TF-IDF and BPE-TF-IDF baselines.
   - Use this notebook to establish reference accuracy before training the transformer.

3. `3_transformer_training.ipynb`
   - Train a Transformer classifier using the proposal-style burst tokens.
   - Inspect the validation accuracy and saved artifacts.

### Using larger datasets

The dataset files represent progressively larger dataset variants. Once the pipeline works with `tor_100w_2500tr.npz`, try switching `DATASET_NAME` in the notebooks to one of:

- `tor_200w_2500tr.npz`
- `tor_500w_2500tr.npz`
- `tor_900w_2500tr.npz`

This will let you compare model behavior and runtime as dataset size increases.

## Recommended next steps

1. Run `1_data_and_tokenization.ipynb` to confirm dataset loading and inspect burst tokens.
2. Run `2_baselines.ipynb` and record baseline accuracy for coarse burst and BPE-based representations.
3. Run `3_transformer_training.ipynb` and save the best model results.
4. Add explicit evaluation on a held-out test split and compute precision/recall/F1.
5. Perform hyperparameter tuning:
   - BPE vocabulary size
   - Transformer depth and number of attention heads
   - Input sequence length (`MAX_LEN`)
   - Learning rate and batch size
6. Compare results across the different `.npz` dataset files.

## Notes

- The notebooks contain a Colab compatibility branch, but the project is structured so it can also run locally.
- The tokenizer and transformer code uses proposal-style burst tokens for richer traffic representation.
- If you want to standardize this work into scripts, the notebook logic can be refactored into Python modules and command-line training scripts.
