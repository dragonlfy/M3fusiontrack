# Architecture Overview

This document is a quick map between the components of the paper and the
modules in this repository. For full mathematical details, see the
corresponding section of the paper.

```
                        ┌─────────────────────────────────────────────────┐
                        │                  Input                          │
                        │  RGB · TIR · Event · Depth   (per modality)     │
                        └─────────────────────────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ LearnableSpectralDecomposition     (models/decomposition.py) │
              │   per modality → B frequency bands (low / mid / high)        │
              │   soft Daubechies-orthonormality penalty (Eq. 3)             │
              └─────────────────────────────────────────────────────────┘
                                            │  (M, B, C, H, W)
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ SharedFoundationBackbone           (models/backbone.py)      │
              │   single ViT + LoRA over all (modality, band) inputs         │
              │   learnable (modality, band) prompts added to CLS (Eq. 4)    │
              └─────────────────────────────────────────────────────────┘
                                            │  (N, M*B, L+1, D)
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ CrossModalFrequencyAttention       (models/fusion.py)        │
              │   attention with learned relative bias φ(k_i, b_i, k_j, b_j) │
              │   over the joint (modality, band) token grid (Eqs. 5–6)      │
              └─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ UncertaintyAwareGating             (models/fusion.py)        │
              │   per-cell score s and variance σ²                           │
              │   g = softmax(s − σ²/τ_g)  fused tokens                      │
              │   spectral consistency regulariser (Eq. 7)                   │
              └─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ FrequencyAwareCorrelation          (models/correlation.py)   │
              │   FFT2(template) · conj(FFT2(search)) · Beta-band mask       │
              │   (Eq. 12)                                                   │
              └─────────────────────────────────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────┐
              │ TrackingHead                       (models/head.py)          │
              │   classification map  +  l/t/r/b regression                  │
              └─────────────────────────────────────────────────────────┘
```

## Training

* `trainer.py` implements the modality-dropout curriculum from §3.7 —
  the probability of dropping one modality ramps from 0 to
  `modality_dropout_max` over the first half of training.
* `utils/losses.py` packages the focal-cls + GIoU + auxiliary
  regulariser terms into a single `M3Loss` object.

## Differences from the paper

This repo is a *simplified* reference: it favours readable code over
maximum performance. Concretely:

* The default backbone is a tiny `SimpleViT` (4 layers, 192-dim) so the
  model runs on a laptop CPU. For real experiments set
  `model.backbone_name: dinov2_vitb14` in the config.
* The learnable spectral decomposition uses smooth 1-D filters rather
  than a strict DWT lifting scheme. The orthonormality penalty is the
  same.
* No long-term template-memory module (the paper's full system pairs
  M3FusionTrack with a HiPTrack-style memory; this repo only does
  short-term tracking).
* Multi-GPU training and AMP are not wired up.
