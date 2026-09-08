# Comparative Study of Automatic Speech Recognition Models

## 1. Introduction

Automatic Speech Recognition (ASR) is the process of converting spoken language into written text.

ASR systems have evolved significantly over the years. Early systems relied on multiple independently designed components such as acoustic models, pronunciation dictionaries, and language models. Modern systems increasingly use end-to-end deep learning and large pretrained models that learn speech representations directly from data.

This document studies the major ASR architectures and compares their strengths, limitations, and potential applications.

---

# 2. Evolution of Automatic Speech Recognition

```text
Traditional Statistical ASR
        │
        ▼
GMM-HMM Systems
        │
        ▼
DNN-HMM Hybrid Systems
        │
        ▼
End-to-End Deep Learning
        │
        ├── CTC
        ├── Sequence-to-Sequence
        └── RNN-T
        │
        ▼
Transformer-Based ASR
        │
        ├── Transformer
        └── Conformer
        │
        ▼
Self-Supervised Learning
        │
        ├── wav2vec 2.0
        ├── HuBERT
        └── WavLM
        │
        ▼
Large-Scale Foundation Models
        │
        ├── XLS-R
        └── Whisper
```

---

# 3. Traditional ASR Models

## 3.1 GMM-HMM

Traditional speech recognition systems commonly used:

* **Gaussian Mixture Models (GMM)** for acoustic modeling.
* **Hidden Markov Models (HMM)** for modeling temporal transitions in speech.

A typical traditional ASR pipeline consisted of:

```text
Speech
  ↓
Feature Extraction (MFCC)
  ↓
Acoustic Model (GMM-HMM)
  ↓
Pronunciation Dictionary
  ↓
Language Model
  ↓
Text Output
```

### Advantages

* Well understood and extensively researched.
* Computationally manageable.
* Effective with carefully engineered features.

### Limitations

* Complex multi-stage architecture.
* Requires manual feature engineering.
* Requires pronunciation dictionaries and language models.
* Difficult to optimize the entire system jointly.

---

## 3.2 DNN-HMM

DNN-HMM systems improved traditional ASR by replacing Gaussian Mixture Models with Deep Neural Networks.

```text
Speech Features
      ↓
Deep Neural Network
      ↓
HMM Sequence Modeling
      ↓
Language Model
      ↓
Text
```

### Advantages

* Improved acoustic modeling.
* Better recognition accuracy than GMM-HMM systems.

### Limitations

* Still depends on multiple independent components.
* Training pipeline remains complex.

---

# 4. End-to-End ASR Models

End-to-end ASR attempts to directly learn the mapping:

```text
Speech → Text
```

This reduces the dependence on manually designed intermediate components.

---

## 4.1 Connectionist Temporal Classification (CTC)

CTC is a training objective used when the alignment between speech frames and output text is unknown.

For example:

```text
Audio Frames:

A  A  _  B  B  _  C

CTC Decoding:

A B C
```

Where `_` represents a blank token.

### Advantages

* No explicit frame-level alignment required.
* Relatively simple architecture.
* Efficient decoding.

### Limitations

* Assumes conditional independence between output predictions.
* External language models may improve performance.

### Examples

* DeepSpeech
* wav2vec 2.0 ASR fine-tuning

---

## 4.2 DeepSpeech and DeepSpeech 2

DeepSpeech was one of the major end-to-end deep learning ASR systems.

DeepSpeech 2 used combinations of:

* Convolutional Neural Networks
* Recurrent Neural Networks
* Connectionist Temporal Classification

### Architecture

```text
Audio Features
      ↓
CNN
      ↓
RNN / LSTM
      ↓
Fully Connected Layers
      ↓
CTC Decoder
      ↓
Text
```

### Advantages

* End-to-end architecture.
* Reduced manual engineering.

### Limitations

* RNN-based architectures are slower for long sequences.
* Older compared to Transformer-based approaches.

---

## 4.3 Listen, Attend and Spell (LAS)

LAS introduced an encoder-decoder architecture with attention.

```text
Speech
   ↓
Encoder
   ↓
Attention Mechanism
   ↓
Decoder
   ↓
Text
```

The attention mechanism allows the decoder to focus on relevant portions of the input audio while generating each output token.

### Advantages

* Learns alignment automatically.
* Strong sequence modeling.

### Limitations

* Can suffer from high inference latency.
* Not naturally optimized for streaming applications.

---

## 4.4 RNN Transducer (RNN-T)

RNN-T is widely used for streaming speech recognition.

It consists of three major components:

```text
Audio
  │
  ▼
Encoder ──────────────┐
                      │
                      ▼
                  Joint Network
                      ▲
                      │
Prediction Network ───┘
                      │
                      ▼
                     Text
```

### Advantages

* Supports streaming ASR.
* Suitable for real-time applications.
* Does not require complete audio before generating output.

### Limitations

* More complex training and decoding.
* Requires careful optimization.

---

# 5. Transformer-Based ASR

## 5.1 Transformer

Transformers use self-attention mechanisms to capture relationships between different parts of a sequence.

Unlike RNNs, Transformers can process sequences in parallel during training.

### Advantages

* Captures long-range dependencies.
* Highly parallelizable.
* Scales effectively with data and model size.

### Limitations

* Self-attention can be computationally expensive for long audio sequences.

---

## 5.2 Conformer

Conformer combines:

* Convolutional Neural Networks
* Transformer Self-Attention

The purpose is to capture both:

* Local acoustic features through convolution.
* Global dependencies through self-attention.

### Simplified Architecture

```text
Input Speech Features
        │
        ▼
Feed Forward Module
        │
        ▼
Multi-Head Self Attention
        │
        ▼
Convolution Module
        │
        ▼
Feed Forward Module
        │
        ▼
Speech Representation
```

### Advantages

* Excellent ASR performance.
* Captures local and global information.
* Widely used in modern speech recognition research.

### Limitations

* More computationally complex.
* Training from scratch requires significant resources.

---

# 6. Self-Supervised Speech Models

Self-supervised learning allows models to learn useful representations from large amounts of unlabeled speech.

The general idea is:

```text
Large Unlabeled Audio Dataset
             ↓
Self-Supervised Pretraining
             ↓
Learned Speech Representations
             ↓
Fine-Tuning with Smaller Labeled Dataset
             ↓
ASR Model
```

This significantly reduces dependence on labeled speech data.

---

## 6.1 wav2vec 2.0

wav2vec 2.0 learns speech representations directly from raw audio.

### Architecture

```text
Raw Audio
    │
    ▼
Feature Encoder
    │
    ▼
Transformer Context Network
    │
    ▼
Self-Supervised Learning
    │
    ▼
Fine-Tuning for ASR
```

### Key Strength

wav2vec 2.0 can leverage large amounts of unlabeled speech data during pretraining.

### Advantages

* Strong performance with limited labeled data.
* Powerful pretrained speech representations.
* Popular for transfer learning.

### Limitations

* Fine-tuning can require significant computational resources.
* Performance depends on language/domain similarity.

---

## 6.2 HuBERT

HuBERT stands for **Hidden-Unit BERT**.

HuBERT learns speech representations by predicting hidden target units from masked audio segments.

### Advantages

* Strong pretrained speech representations.
* Effective for speech recognition and related tasks.

### Limitations

* Large models require substantial computation.
* Fine-tuning is generally required for task-specific deployment.

---

## 6.3 WavLM

WavLM is a self-supervised speech representation model designed to learn robust representations.

It supports multiple speech tasks including:

* Automatic Speech Recognition
* Speaker Recognition
* Speech Understanding

### Advantages

* Robust speech representations.
* General-purpose speech model.

### Limitations

* Large computational requirements for bigger variants.

---

# 7. Multilingual ASR

## XLS-R

XLS-R is a multilingual self-supervised speech representation model.

It extends cross-lingual learning to a large number of languages.

### Advantages

* Strong multilingual capability.
* Useful for low-resource languages.
* Suitable for cross-lingual transfer learning.

### Limitations

* Large models require substantial memory and computation.
* Fine-tuning may be required for optimal task performance.

---

# 8. Whisper

Whisper is a large-scale pretrained Automatic Speech Recognition model based on the encoder-decoder Transformer architecture.

### Simplified Pipeline

```text
Audio
  │
  ▼
Log-Mel Spectrogram
  │
  ▼
Audio Encoder
  │
  ▼
Transformer Decoder
  │
  ▼
Text Tokens
```

### Capabilities

Whisper supports:

* Speech transcription.
* Multilingual speech recognition.
* Speech translation.
* Language identification.

### Advantages

* Strong pretrained baseline.
* Multilingual support.
* Robustness to accents and noisy audio.
* Easy to use through pretrained implementations.

### Limitations

* Larger models require significant computational resources.
* Inference can be slower than lightweight ASR models.

---

# 9. Comprehensive Model Comparison

| Model        | Architecture                | Training Approach       | Strength                     | Limitation                | Suitable Use          |
| ------------ | --------------------------- | ----------------------- | ---------------------------- | ------------------------- | --------------------- |
| GMM-HMM      | Statistical                 | Supervised              | Well understood              | Complex pipeline          | Traditional ASR       |
| DNN-HMM      | DNN + HMM                   | Supervised              | Improved acoustics           | Multi-stage system        | Hybrid ASR            |
| DeepSpeech 2 | CNN + RNN + CTC             | Supervised              | End-to-end                   | Older architecture        | Learning/Baseline     |
| LAS          | Encoder-Decoder + Attention | Supervised              | Strong sequence modeling     | High latency              | Offline ASR           |
| RNN-T        | Transducer                  | Supervised              | Streaming                    | Complex training          | Real-time ASR         |
| Transformer  | Self-Attention              | Supervised              | Long-range dependencies      | High computation          | Modern ASR            |
| Conformer    | CNN + Transformer           | Supervised              | Local + global modeling      | Complex                   | High-performance ASR  |
| wav2vec 2.0  | Transformer                 | Self-Supervised         | Low labeled-data requirement | Fine-tuning required      | Transfer Learning     |
| HuBERT       | Transformer                 | Self-Supervised         | Strong representations       | Computationally intensive | Speech Representation |
| WavLM        | Transformer                 | Self-Supervised         | Robust general features      | Large models              | Multiple Speech Tasks |
| XLS-R        | Transformer                 | Self-Supervised         | Multilingual                 | High compute              | Multilingual ASR      |
| Whisper      | Encoder-Decoder Transformer | Large-scale Pretraining | Robust multilingual ASR      | Computationally intensive | General ASR           |

---

# 10. Comparison Based on Important Parameters

## 10.1 Accuracy

Modern pretrained models such as Whisper, wav2vec 2.0, and Conformer-based systems generally outperform older architectures when sufficient training data and computational resources are available.

---

## 10.2 Streaming Capability

Models suitable for real-time streaming include:

* RNN-T
* Streaming Conformer architectures

Whisper is primarily designed for processing chunks of audio rather than being inherently streaming.

---

## 10.3 Multilingual Capability

Strong multilingual candidates include:

* Whisper
* XLS-R

These models are particularly relevant when the target application involves multiple languages or low-resource languages.

---

## 10.4 Computational Requirements

| Category  | Models                             |
| --------- | ---------------------------------- |
| Lower     | GMM-HMM, smaller DeepSpeech models |
| Moderate  | wav2vec 2.0 base                   |
| High      | Conformer, HuBERT, WavLM           |
| Very High | Large XLS-R and Whisper models     |

---

# 11. Candidate Models for Aksharvel

Based on the initial study, the following models are shortlisted for further experimentation.

## 1. Whisper

### Why?

* Strong pretrained baseline.
* Robust speech recognition.
* Multilingual capability.
* Easy initial experimentation.

---

## 2. wav2vec 2.0 / XLS-R

### Why?

* Self-supervised learning.
* Useful when labeled data is limited.
* Strong potential for fine-tuning on domain-specific data.

---

## 3. Conformer

### Why?

* Modern high-performance architecture.
* Strong combination of local acoustic modeling and global attention.

---

# 12. Recommended Experimental Approach

The recommended development process is:

```text
Study Dataset
     ↓
Understand Audio Characteristics
     ↓
Select Baseline Model
     ↓
Run Pretrained Model Inference
     ↓
Evaluate WER / CER
     ↓
Analyze Errors
     ↓
Fine-Tune if Required
     ↓
Compare Candidate Models
```

---

# 13. Conclusion

Automatic Speech Recognition has evolved from traditional statistical pipelines to powerful end-to-end and pretrained foundation models.

For the Aksharvel project, it is not practical to train a large ASR system from scratch initially. A more effective approach is to begin with pretrained models and establish baseline performance.

The recommended initial models are:

1. **Whisper** as a robust pretrained baseline.
2. **wav2vec 2.0 / XLS-R** for self-supervised and multilingual experimentation.
3. **Conformer** for studying modern high-performance ASR architectures.

The next stage of work is to experimentally evaluate these models on the target dataset and compare their performance using Word Error Rate, Character Error Rate, latency, and computational requirements.

