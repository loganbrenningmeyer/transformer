set -ex

CONFIG=$1
python -m transformer.train_seq2seq --config "$CONFIG"