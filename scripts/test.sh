set -ex

CONFIG=$1
python -m transformer.test --config "$CONFIG"