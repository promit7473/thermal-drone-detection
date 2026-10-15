#!/bin/bash
cd "$(dirname "$0")"
python3 src/thermal_detect.py "$@"
