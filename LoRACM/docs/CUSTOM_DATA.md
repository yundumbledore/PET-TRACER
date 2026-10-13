# Adapt PET-TRACER to your own data

[User manual](../README.md) · [Models and protocols](MODELS_AND_PROTOCOLS.md) · [Prepare a labelled dataset](PREPARE_DATA.md) · [Prediction](PREDICTION.md)

Use the portable protocol workflow for new tracers, custom kinetic parameter sets, explicit frame durations and midpoint/frame-average observations. You can simulate labelled pairs or validate your own NPZ and skip generation. Both the GUI and terminal workflows are documented in the manual.

The historical `create_dataset.py` and Scenario 1 configs retain their FDG–2TC forward-Euler assumptions; see the [UC Davis guide](UCDAVIS.md) to access that path. A different AIF alone does not establish support or scientific validation for another tracer/model.
