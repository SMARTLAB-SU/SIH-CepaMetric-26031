# Onion AI Inspection System

Smart India Hackathon repository for an automated onion inspection system that combines deep-learning disease classification, Intel RealSense depth sensing, classical object tracking, physical size measurement, explainable AI, batch counting, and Arduino-based conveyor control.

> Repository placeholder: replace `TeamName` and `SIHID` before submission.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Proposed Solution](#proposed-solution)
4. [Core Features](#core-features)
5. [System Architecture](#system-architecture)
6. [Classification Classes](#classification-classes)
7. [Models](#models)
8. [Dataset](#dataset)
9. [Hardware](#hardware)
10. [Repository Structure](#repository-structure)
11. [Installation](#installation)
12. [Running the System](#running-the-system)
13. [Testing](#testing)
14. [Explainable AI Evidence](#explainable-ai-evidence)
15. [Results and Current Evidence](#results-and-current-evidence)
16. [Installer and Deployment Status](#installer-and-deployment-status)
17. [Known Limitations](#known-limitations)
18. [Team Information](#team-information)
19. [License](#license)

## Overview

The Onion AI Inspection System is a PyQt5 desktop application designed for live agricultural quality inspection. It processes camera frames, tracks individual onions, estimates physical size using depth information, classifies visible condition, records batch counts, and can produce Grad-CAM visual explanations.

The application supports five TensorFlow/Keras base-model architectures:

- ConvNeXt Tiny
- EfficientNetV2-B0
- DenseNet121
- ResNet50V2
- Xception

It also provides a stacking meta-classifier and an ensemble mode that averages probabilities from the five base models.

The supplied project was reorganized into a minimal SIH repository. Local virtual environments, cache files, duplicate weights, and raw datasets were not copied. The original project folder remains unchanged.

## Problem Statement

Manual onion inspection can be slow, inconsistent, and difficult to scale on a conveyor or packing line. Inspectors may need to assess visible disease condition, count individual onions without duplication, estimate size, assign a grade, and maintain batch-level records while products continue moving.

An automated system must combine reliable image classification with stable physical tracking and measurement. It must also expose sufficient evidence for operators to review incorrect or uncertain decisions.

## Proposed Solution

This project combines four functional layers:

1. **Image acquisition:** RGB/depth capture from an Intel RealSense D455f or simulated input.
2. **Tracking and measurement:** classical OpenCV geometry, persistent IDs, depth sampling, and physical-size estimation.
3. **Quality classification:** TensorFlow/Keras models predict one of four onion-condition classes.
4. **Operator interface:** a PyQt5 dashboard shows the live feed, model selection, counters, history, size/grade information, and Grad-CAM explanations.

An Arduino sketch is included for servo/conveyor integration. The main application currently expects the controller on `COM3` at 9600 baud.

## Core Features

- Live camera and simulated-input workflows.
- Five selectable base classification models.
- Stacking meta-classifier using a 20-value probability vector.
- Five-model probability-averaging ensemble.
- Four-class onion condition classification.
- Persistent object IDs to reduce duplicate counting.
- Depth-assisted physical diameter estimation.
- Batch number and slot quantity workflow.
- Automatic slot-completion handling.
- Disease and size counters.
- Capture-history navigation.
- On-demand Grad-CAM generation.
- Responsive inference/XAI worker threads.
- SMART Lab application branding and window icon.
- Arduino serial commands for external actuation.

## System Architecture

```text
RealSense D455f / camera / simulator
                  │
                  ▼
       Classical detection and tracking
                  │
         ┌────────┴────────┐
         ▼                 ▼
 Depth-based sizing    RGB onion crop
         │                 │
         │                 ▼
         │       Selected Keras classifier
         │                 │
         └────────┬────────┘
                  ▼
       PyQt5 dashboard and counters
                  │
          ┌───────┴────────┐
          ▼                ▼
    Grad-CAM review    Arduino command
```

Default source settings found in the application:

| Setting | Default |
| --- | --- |
| Arduino port | `COM3` |
| Arduino baud rate | `9600` |
| Camera index | `0` |
| Capture interval | `4000 ms` |
| Conveyor depth | `850 mm` |
| Default selected model | DenseNet121 |

These values must be calibrated for the deployment environment.

## Classification Classes

The application predicts four classes:

| Application label | Notebook label | Intended category |
| --- | --- | --- |
| Moldy | `molded` | Onion showing visible mold-related deterioration |
| Normal | `normal` | Onion without the targeted visible defects |
| Rotten | `rotten` | Onion showing visible rot or decay |
| Sprouted | `sprouted` | Onion showing visible sprouting |

Descriptions are practical summaries only. Confirm the authoritative class definitions and labeling policy from the dataset owner.

## Models

### Model Layout

```text
Model/Classification/
├── ConvNeXtTiny/
│   └── convnexttiny_model.weights.h5
├── EfficientNetV2B0/
│   └── effnetv2b0model.weights.h5
├── DenseNet121/
│   └── onion_densenet121_best_macro_f1.weights.h5
├── ResNet50V2/
│   └── resnet_model.weights.h5
├── Xception/
│   ├── xception_model.weights.h5
│   └── onion_kaggle_xception_300epochs_balanced.ipynb
└── Stacking/
    └── onion_stacking_meta_classifier_best.keras
```

The application builds each base architecture in code and loads the corresponding weights on demand. Xception uses a 299×299 input; the other base models use 224×224 input. Each base model produces four class probabilities.

The stacking model consumes the five four-class probability vectors in the application's fixed base-model order. Ensemble mode averages those five vectors directly and therefore has no separate checkpoint.

Only the Xception training notebook was present. It describes 300 epochs, ImageNet transfer learning, class weights, optional focal loss, AdamW, validation Macro-F1 checkpointing, balanced accuracy, per-class metrics, and confusion-matrix analysis.

## Dataset

No raw dataset was copied into this repository. The supplied Xception notebook references [Onion Dataset Combined](https://www.kaggle.com/datasets/yashraneja2/onion-dataset-combined) and the classes `molded`, `normal`, `rotten`, and `sprouted`.

See [Dataset/README.md](Dataset/README.md) for access and verification requirements.

## Hardware

### Intel RealSense D455f

The depth subsystem includes source code for:

- RGB/depth acquisition.
- Camera-device inspection.
- Classical contour and edge analysis.
- Persistent multi-object tracking.
- Depth sampling and metric conversion.
- Simulator-based offline operation.
- CSV measurement output.

The included detection subsystem was originally documented as a non-AI box-measurement system. Its suitability for onion geometry requires dedicated validation.

### Arduino

`SRC/Hardware/arduino_servo_control.ino` contains the supplied servo-control sketch. Upload it with the Arduino IDE and update `ARDUINO_PORT` in the main application if the controller is not assigned to `COM3`.

## Repository Structure

```text
SIH-Onion-Classification-TeamName-SIHID/
├── Dataset/
│   └── README.md
├── Model/
│   └── Classification/
│       ├── ConvNeXtTiny/
│       ├── EfficientNetV2B0/
│       ├── DenseNet121/
│       ├── ResNet50V2/
│       ├── Xception/
│       └── Stacking/
├── SRC/
│   ├── Classification_App/
│   ├── Detection_App/
│   ├── Hardware/
│   ├── Logo/
│   ├── Installer/
│   └── Needs_Review/
├── Screenshot/
│   ├── Screenshots/
│   └── Demo_Video/
├── Documentation/
├── .gitattributes
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

Only the main README and Dataset README are used. Unnecessary folder-level README files were not copied.

## Installation

### Prerequisites

- Windows 10/11 or Linux with compatible RealSense support.
- Python version compatible with TensorFlow 2.16 and the RealSense Python package.
- Intel RealSense SDK/runtime for live D455f operation.
- Arduino IDE if the servo controller is used.
- Git LFS for downloading and publishing the large model files.

### Python Environment

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS activation:

```bash
source .venv/bin/activate
```

### Git LFS

The model files exceed normal GitHub file-size limits and are configured for Git LFS.

```bash
git lfs install
git lfs pull
```

## Running the System

### Integrated Onion Inspection Application

From the repository root:

```bash
python SRC/Classification_App/main.py
```

The application attempts to connect to the configured Arduino port during startup. If no controller is attached, it reports a warning and continues.

### RealSense Detection and Measurement Application

With a connected D455f camera:

```bash
python SRC/Detection_App/main.py
```

Using the simulator:

```bash
python SRC/Detection_App/main.py --simulate
```

Windows helper launchers are also retained:

```text
SRC/Detection_App/run_gui.bat
SRC/Detection_App/run_camera_test.bat
```

They use the repository-level `.venv` when present and otherwise fall back to the system `python` command.

## Testing

Run tests from the detection-application directory so its local imports resolve correctly:

```bash
cd SRC/Detection_App
python test_conveyor_box.py
python test_detection.py
python test_onion_detection.py
```

Camera verification:

```bash
python SRC/Detection_App/camera_test.py
```

Hardware-dependent tests require a compatible RealSense installation and may need calibration or local configuration.

## Explainable AI Evidence

`Screenshot/Screenshots/XAI_GradCAM/` contains three supplied Integrated Grad-CAM overlays, their heatmap arrays, and a CSV index. These are model-explanation artifacts, not application-interface screenshots.

The source file named `xai_manifest.json` was not valid JSON; it contained NumPy binary data. It is preserved under `SRC/Needs_Review/xai_manifest_mislabeled.npy` for manual inspection.

## Results and Current Evidence

No complete evaluation report or executed notebook metrics were found. The available XAI index contains three Normal-class examples:

| Test index | True class | Predicted class | Confidence |
| ---: | --- | --- | ---: |
| 130 | normal | rotten | 77.64% |
| 51 | normal | normal | 99.88% |
| 68 | normal | normal | 99.88% |

This small selection must not be treated as an accuracy estimate. Recompute test accuracy, Macro-F1, balanced accuracy, per-class precision/recall/F1, and a confusion matrix using the confirmed dataset split.

## Installer and Deployment Status

No installer, packaged executable, PyInstaller definition, or Inno Setup script was found. `SRC/Installer/` is reserved for a future tested installer.

Before deployment:

1. Validate all relative model and asset paths.
2. Confirm TensorFlow and RealSense runtime packaging.
3. Bundle the SMART Lab icon and logo.
4. Test camera, Arduino, and filesystem permissions.
5. Test installation and uninstallation on a clean computer.

## Known Limitations

- Dataset statistics and license details require confirmation.
- Overall model metrics were not supplied.
- Training code is missing for five retained trained components.
- Application/notebook label spelling differs for the Moldy/molded class.
- One source XAI file had an incorrect `.json` extension.
- RealSense sizing accuracy depends on calibration and scene geometry.
- The depth subsystem documentation focuses on boxes rather than onions.
- Hardware defaults are currently hard-coded.
- No installer, formal SIH report, SIH presentation, UI screenshots, or demo video was found.
- Model weights and XAI outputs require ownership and publication-rights review.

See `SRC/Needs_Review/MANUAL_REVIEW.md` for the complete submission checklist.

## Team Information

- **Team Name:** `TeamName` — replace before submission
- **SIH ID:** `SIHID` — replace before submission
- **Institution:** To be added
- **Team Members:** To be added
- **Project Maintainer:** To be added
- **Contact:** To be added

## License

The repository includes an MIT License. Confirm ownership and redistribution permission for the source, model weights, logo, dataset-derived outputs, and third-party components before public release.
