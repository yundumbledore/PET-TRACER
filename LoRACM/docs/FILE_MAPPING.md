# Research files → release module

| Research file | Release location | Treatment |
| :--- | :--- | :--- |
| `Create_dataset.py` | `simulation.py` + `create_dataset.py` + `configs/dataset_*.json` | Original simulation functions retained; portable measured-AIF driver and bounded outer chunks added |
| `UCDavis_1H35.csv` | `data/aif/UCDavis_1H35.csv` | Copied unchanged |
| `train_lora_cm.py` | `train.py` + `configs/train_*.json` | Objective retained; configuration, validation and output handling improved |
| `inference_realdata_mgpus.py` | `infer.py` | Sampling procedure retained; CLI, memory control, correct summaries and data checks added |
| `Parametric imaging viewer.ipynb` | `notebooks/01_parametric_imaging.ipynb` | Replaced exploratory cells with a guided, verified mapping and export workflow |
| `prepare_dataset.py`, `model.py`, `lora.py`, `utility.py` | Same names inside `LoRACM/` | Supporting files copied from the local research folder |
| `model_lora.py` | `model_lora.py` | Compatible-weight transfer retained; missing-weight training and exact-name matching checked |
| Original repository `README.md` | `docs/legacy-framework.md` | Preserved with image links adjusted for its new location |
