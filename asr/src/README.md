# ASR Source Code

This directory will contain reusable Python modules for the Automatic Speech Recognition pipeline.

## Planned Modules

```text
src/
├── preprocessing.py
├── inference.py
├── evaluation.py
└── utils.py
```

### preprocessing.py

Functions related to:

* Audio loading.
* Resampling.
* Audio normalization.
* Feature preparation.

### inference.py

Functions for:

* Loading pretrained ASR models.
* Running speech recognition inference.
* Processing transcription outputs.

### evaluation.py

Functions for evaluating model performance using:

* Word Error Rate (WER).
* Character Error Rate (CER).

### utils.py

Common utility functions used across experiments.

