set -ex

CONFIG=$1
python -m transformer.train --config "$CONFIG"