
# AGENT DIRECTIVE: Repository Modernization & Academic Refactoring

**Target Repository:** `Vietnamese_sign_language_recognition_using_MHI_and_CNN`
**Author:** Phạm Hồng Phát
**Context:** HCMUT Capstone Thesis (**Grade: 9.17 / 10.0** | B.Eng. in Engineering Physics)
**Primary Audience:** Admissions Committees & Technical Reviewers for Erasmus Mundus Joint Masters:

- **DEAI (Dependable AI):** Evaluates resource-constrained Edge AI, latency/memory profiling, hardware-software co-design, and execution safety.
- **EDISS (Data-Intensive Software Systems):** Evaluates software architecture, clean code principles, modular package structure, and CLI workflow engineering.
- **CoDAS (Communications, Signals & Data Science):** Evaluates spatial-temporal signal transformation (MHI), mathematical formulation, and computer vision feature extraction.

---

## EXECUTIVE GOAL

Transform this historical capstone repository from an undergraduate-era project into an **auditable, production-grade, and academically rigorous open-source Edge AI benchmark**.

You must execute this refactoring in two distinct, sequential phases:

1. **Phase 1: `README.md` Overhaul** (Primacy-driven, cognitively ergonomized visual and academic front-page).
2. **Phase 2: Source Code Architecture & Package Refactoring** (Modularizing, typing, modernizing, and benchmarking the codebase).

---

## PHASE 1: `README.md` OVERHAUL

Replace the existing `README.md` completely with the following structured Markdown. Ensure all mathematical equations, diagrams, and benchmark tables are rendered cleanly.

### `README.md` Specification

````markdown
# Real-Time Edge AI Sign Language Recognition using MHI + 2D-CNN

[![Academic Verification](https://img.shields.io/badge/HCMUT%20Thesis%20Grade-9.17%20%2F%2010.0-blue.svg)](#academic-verification)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Target Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%203B%2B%20%2F%204-red.svg)](#hardware--performance-benchmarks)

> **Abstract:** A resource-efficient, real-time spatial-temporal gesture recognition pipeline designed for embedded Edge AI execution. By compressing video frame time-series into single-channel **Motion History Images (MHI)** prior to spatial classification via a lightweight 2D Convolutional Neural Network (CNN), this system achieves real-time inference on resource-constrained single-board computers (Raspberry Pi) without requiring heavy 3D-CNNs or GPU acceleration.

---

## 1. Academic Verification & Context

* **Thesis Title:** Vietnamese Sign Language Recognition using MHI and CNN on Embedded Systems
* **Institution:** Ho Chi Minh City University of Technology (HCMUT)
* **Degree Program:** B.Eng. in Engineering Physics (Biomedical Engineering Specialization)
* **Capstone Evaluation Grade:** **9.17 / 10.0** (Evaluated by Academic Council)
* **Author:** Phạm Hồng Phát ([phatpham.work](https://phatpham.work))

---

## 2. System Architecture

[ Camera Stream ] ──> [ Temporal Motion Filtering ] ──> [ MHI Generation ]
│
▼
[ Real-Time Inference ] <── [ Lightweight 2D-CNN ] <── [ Spatial Crop ]


### Mathematical Formulation: Motion History Image (MHI)

Temporal motion is encoded into a single 2D spatial representation where pixel intensity decay represents motion recency:

$$H_{\tau}(x, y, t) = \begin{cases} \tau & \text{if } \Psi(x, y, t) = 1 \\ \max(0, H_{\tau}(x, y, t - 1) - 1) & \text{otherwise} \end{cases}$$

Where:
* $\Psi(x, y, t)$ is a binary frame-difference mask calculated via $\lvert I(x,y,t) - I(x,y,t-1) \rvert > \xi$.
* $\tau$ represents the temporal history window duration (decay parameter).
* $H_{\tau}(x, y, t)$ yields a grayscale image matrix where bright pixels represent recent movement and darker gradients encode trajectory history.

---

## 3. Hardware & Performance Benchmarks

Evaluated on embedded single-board targets under thermal and memory constraints:

| Metric | Embedded Target (Raspberry Pi 3B+) | Development Host (x86_64) |
| :--- | :--- | :--- |
| **Target OS / Environment** | Raspberry Pi OS (Debian armhf) | Linux (x86_64 Ubuntu 22.04) |
| **Inference Engine** | TFLite / OpenCV DNN | TensorFlow 2.x Keras |
| **Frame Preprocessing (MHI)** | ~4.2 ms / frame | ~0.8 ms / frame |
| **Model Inference Latency** | ~28.5 ms / frame | ~3.1 ms / frame |
| **End-to-End Pipeline FPS** | **~28--30 FPS (Real-Time)** | **>120 FPS** |
| **Peak RAM Footprint** | < 180 MB | < 450 MB |
| **Gesture Classification Accuracy** | **94.2%** (Validation Set) | **94.2%** |

---

## 4. Key Engineering & Dependable AI Features

* **Resource-Bounded Compute:** Replaces compute-heavy 3D-CNN spatio-temporal convolutions with $O(1)$ temporal MHI image buffer accumulation, lowering memory footprint by over 70%.
* **Determinism & Low Latency:** Optimized image transformation pipeline written in C++-backed OpenCV primitives for real-time camera stream consumption.
* **Modular Infrastructure:** Separated dataset processing, MHI feature extraction, model definition, and live camera inference engines into clean CLI interfaces.

---

## 5. Repository Structure

```text
.
├── configs/                # System & Model Hyperparameter Configurations
│   └── default_config.yaml
├── data/                   # Dataset Directory Structure (Git Ignored)
│   ├── raw/
│   └── processed_mhi/
├── models/                 # Model Architecture & Saved Weights
│   └── cnn_mhi_model.h5
├── src/                    # Core Python Package
│   ├── __init__.py
│   ├── dataset.py          # Data Loading & Motion Masking
│   ├── mhi.py              # Motion History Image Transformation Engine
│   ├── model.py            # Keras 2D-CNN Architecture Definition
│   └── utils.py            # Profiling & Visualization Helpers
├── scripts/                # Execution Scripts
│   ├── benchmark.py        # Latency & Memory Footprint Profiler
│   ├── train.py            # Model Training & Validation Pipeline
│   └── live_inference.py   # Real-Time Webcam/PiCamera Edge Pipeline
├── requirements.txt        # Verified Dependency Manifest
└── README.md
````
