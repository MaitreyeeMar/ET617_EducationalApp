# Automatic Speech Recognition (ASR) Research

This directory contains the research, model exploration, and implementation work related to Automatic Speech Recognition (ASR).

## Objective

The objective is to study modern speech recognition architectures and identify suitable models for the Aksharvel speech recognition task.

The work currently focuses on:

* Understanding the ASR pipeline
* Studying major ASR architectures
* Comparing pretrained speech recognition models
* Establishing baseline inference pipelines
* Evaluating models using standard ASR metrics

---

# ASR Pipeline

```text
Speech Audio
     │
     ▼
Audio Preprocessing
     │
     ▼
Feature Extraction / Speech Representation
     │
     ▼
ASR Model
     │
     ▼
Token Prediction / Decoding
     │
     ▼
Text Transcription
```

---

# ASR Models Studied

## Traditional ASR

### GMM-HMM

Traditional speech recognition systems used Gaussian Mixture Models (GMMs) for acoustic modeling and Hidden Markov Models (HMMs) for temporal sequence modeling.

### DNN-HMM

Deep Neural Networks replaced GMMs for improved acoustic modeling while retaining the HMM-based framework.

---

# End-to-End ASR Models

## DeepSpeech / DeepSpeech 2

DeepSpeech introduced end-to-end speech recognition using deep neural networks and Connectionist Temporal Classification (CTC).

Key features:

* End-to-end architecture
* CTC loss
* CNN/RNN-based acoustic modeling

---

## Listen, Attend and Spell (LAS)

LAS uses an encoder-decoder architecture with an attention mechanism.

Architecture:

```text
Audio → Encoder → Attention → Decoder → Text
```

---

## RNN Transducer (RNN-T)

RNN-T is designed for sequence-to-sequence speech recognition and is widely used in streaming ASR systems.

Key advantage:

* Suitable for real-time speech recognition.

---

# Transformer-Based ASR

## Transformer

Transformers use self-attention mechanisms to capture long-range dependencies in speech.

Advantages:

* Parallel processing
* Strong sequence modeling capability

---

## Conformer

The Conformer architecture combines:

* Convolutional Neural Networks
* Transformer self-attention

This allows the model to capture both local acoustic features and long-range dependencies.

---

# Self-Supervised Speech Models

## wav2vec 2.0

wav2vec 2.0 learns speech representations from large amounts of unlabeled audio.

The pretrained model can later be fine-tuned using labeled speech data.

Key advantage:

* Requires significantly less labeled data for downstream tasks.

---

## HuBERT

HuBERT learns hidden speech representations using masked prediction techniques.

It provides powerful pretrained representations for speech-related tasks.

---

## WavLM

WavLM extends self-supervised speech representation learning and focuses on robust speech processing.

Applications include:

* Speech recognition
* Speaker recognition
* Speech understanding

---

# Multilingual ASR

## XLS-R

XLS-R is a multilingual extension of wav2vec-style speech representation learning.

It is designed for cross-lingual and low-resource language speech recognition.

---

# Large-Scale ASR Models

## Whisper

Whisper is a large-scale multilingual Automatic Speech Recognition model.

Capabilities include:

* Speech transcription
* Multilingual recognition
* Speech translation
* Robust recognition under noisy conditions

Whisper is currently being considered as one of the primary baseline models for experimentation.

---

# Initial Model Comparison

| Model        | Architecture                | Strength                         | Limitation                |
| ------------ | --------------------------- | -------------------------------- | ------------------------- |
| DeepSpeech 2 | CNN + RNN + CTC             | Simple end-to-end ASR            | Older architecture        |
| wav2vec 2.0  | Self-Supervised Transformer | Strong with limited labeled data | Fine-tuning required      |
| HuBERT       | Self-Supervised Transformer | Strong speech representations    | Computationally intensive |
| WavLM        | Self-Supervised Transformer | Robust speech understanding      | Large model               |
| XLS-R        | Multilingual SSL            | Good multilingual performance    | High compute requirements |
| Conformer    | CNN + Transformer           | Strong modern architecture       | Training complexity       |
| RNN-T        | Transducer                  | Streaming ASR                    | Complex training          |
| Whisper      | Encoder-Decoder Transformer | Robust multilingual ASR          | Computationally expensive |

---

# Evaluation Metrics

The following metrics will be used for evaluating ASR models.

## Word Error Rate (WER)

WER measures the difference between predicted transcription and reference transcription.

WER is defined as:

```text
WER = (S + D + I) / N
```

Where:

* S = Number of substitutions
* D = Number of deletions
* I = Number of insertions
* N = Number of words in the reference transcription

---

## Character Error Rate (CER)

CER evaluates transcription errors at the character level.

This is particularly useful for languages or applications where word boundaries may not be sufficient for evaluation.

---

# Current Progress

* Studied the overall Automatic Speech Recognition pipeline.
* Reviewed the evolution of ASR architectures.
* Studied traditional, end-to-end, Transformer-based, and self-supervised ASR models.
* Identified Whisper, wav2vec 2.0/XLS-R, and Conformer-based architectures as primary candidates for further investigation.
* Established the initial repository structure for ASR experimentation.

---

# Next Steps

1. Finalize the target dataset.
2. Implement audio preprocessing.
3. Establish a baseline ASR model.
4. Perform inference on sample audio.
5. Fine-tune selected models if required.
6. Evaluate performance using WER and CER.
7. Compare models based on:

   * Recognition accuracy
   * Computational requirements
   * Inference latency
   * Dataset requirements
   * Language support

---

# Repository Structure

```text
asr/
├── docs/          # Research documentation and literature review
├── notebooks/     # Model experiments
├── src/           # ASR source code
└── results/       # Experimental results
```

