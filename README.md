# TimeDeCODE4ClS

A standalone time-series classification pipeline: **standard VQ-VAE → masked-token pretraining → supervised classification**.

The VQ-VAE uses a shared, single-level discrete codebook. A bidirectional Transformer predicts masked code IDs, then its representations feed a classification head. Classification fine-tunes the token embedding, Transformer and head; the VQ-VAE is frozen and its tokens are cached. All three stages are trained separately on each dataset from scratch.

## Setup

```bash
conda create -n time_decode python=3.10.8 -y
conda activate time_decode
pip install -r requirements.txt
```

Test target: NVIDIA H20, PyTorch 2.7.1 with CUDA 12.6. A compatible NVIDIA driver is required. The dependency file pins PyTorch but does not install a driver or guarantee bitwise results on other hardware.

## Data

Place the official [UEA multivariate datasets](https://www.timeseriesclassification.com/dataset.php) as follows. Data and model weights are not distributed in this repository.

```text
datasets/UEA/JapaneseVowels/JapaneseVowels_TRAIN.ts
datasets/UEA/JapaneseVowels/JapaneseVowels_TEST.ts
```

Supported datasets: EthanolConcentration, FaceDetection, Handwriting, Heartbeat, JapaneseVowels, PEMS-SF, SelfRegulationSCP1, SelfRegulationSCP2, SpokenArabicDigits and UWaveGestureLibrary. The reader supports non-timestamped `.ts` files, including unequal-length sequences and missing values.

## Run

```bash
# Short end-to-end check
bash scripts/smoke.sh

# All ten datasets, all three training stages
bash scripts/uea10.sh

# One dataset or an external data directory
DATA_ROOT=/path/to/UEA OUTPUT_ROOT=results/run2 bash scripts/uea10.sh --datasets JapaneseVowels

# Model unit tests
python -m unittest discover -s tests -v
```

Use a new output directory for every experiment; existing dataset result directories are never overwritten. `python train.py --help` lists training options. Reduce `--batch-size` if GPU memory is limited, especially for PEMS-SF.

## Evaluation protocol

- Seed 42. Split the official TRAIN set into stratified training/validation subsets (80/20 by default).
- Fit normalization and select sequence length from the training subset only. Missing values are interpolated within each channel; each sequence is resampled to a common patch-aligned length, capped at 512 by default. This preprocessing may differ from other published protocols.
- VQ-VAE is selected by validation reconstruction-plus-VQ loss. Mask pretraining uses fixed validation masks and selects by masked-token cross-entropy. Neither stage trains on validation or TEST samples.
- Classification selects validation accuracy, with validation cross-entropy as the tie-breaker. TEST is evaluated once after selection, not for early stopping.
- Compare published numbers only when split, preprocessing and selection protocols agree. The scripts are an initial recipe, not a claim of state-of-the-art accuracy or exact final-result reproducibility.

Each run saves parameters, package versions, split indices, normalization, data hashes, epoch metrics and all three stage checkpoints. `result.json` contains final classification results; `summary.json` reports the mean over completed datasets and the completed/requested counts. A full UEA10 average requires ten completed datasets. Training checkpoints are stage artifacts, not full optimizer/RNG resume snapshots.

## Verification status

The two model unit tests and `scripts/smoke.sh` passed on H20 / PyTorch 2.7.1+cu126. The smoke run on JapaneseVowels (1 VQ-VAE epoch, 1 masked-pretraining epoch, 2 classification epochs) obtained 84.05% TEST accuracy with seed 42. This is a pipeline check, not a tuned benchmark. Full ten-dataset results are pending.
