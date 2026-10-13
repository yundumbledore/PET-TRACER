#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ -x .venv/bin/python ]; then
    .venv/bin/python launch_gui.py
else
    python3 launch_gui.py
fi
