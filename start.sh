#!/bin/bash
set -ex

yes | uv run ray start --head --block &
sleep 10  # Give Ray time to start
uv run serve deploy serve_config.yaml
wait