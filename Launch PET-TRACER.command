#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ -n "${PET_TRACER_PYTHON:-}" ]; then
    "$PET_TRACER_PYTHON" launch_gui.py
elif [ -x "$HOME/.venvs/pet-tracer/bin/python" ]; then
    "$HOME/.venvs/pet-tracer/bin/python" launch_gui.py
elif [ -x .venv/bin/python ]; then
    .venv/bin/python launch_gui.py
else
    python3 launch_gui.py
fi
