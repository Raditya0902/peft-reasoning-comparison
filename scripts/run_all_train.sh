#!/usr/bin/env bash
set -euo pipefail
python -m src.train --config configs/lora.yaml
python -m src.train --config configs/dora.yaml
python -m src.train --config configs/ia3.yaml
