set -ex

CONFIG=$1
python -m transformer.train_lm --config "$CONFIG"