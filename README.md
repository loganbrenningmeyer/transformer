# Getting Started

## Installation
* Clone this repo:
```
git clone https://github.com/loganbrenningmeyer/transformer.git
cd transformer
```

* Install dependencies or create/activate a new Conda environment with:
```
conda env create -f environment.yml
conda activate transformer
```

## Setting up Weights & Biases
* Login to wandb / set environment variables:
```
wandb login

export WANDB_ENTITY="<entity>"
export WANDB_PROJECT="<project>"
```

## TransformerLM (Decoder-only)

### Training
* Configure `configs/train_lm.yml` and launch train script:
```
sh scripts/train_lm.sh
```

### Testing
* Configure `configs/test_lm.yml` and launch test script:
```
sh scripts/test_lm.sh
```

## TransformerSeq2Seq (Sequence-to-Sequence)

### Training
* Configure `configs/train_seq2seq.yml` and launch train script:
```
sh scripts/train_seq2seq.sh
```

### Testing
* Configure `configs/test_seq2seq.yml` and launch test script:
```
sh scripts/test_seq2seq.sh
```
