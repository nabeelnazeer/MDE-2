# DynoV2 Teacher Pretraining

This directory contains the code for pretraining the DynoV2 teacher model for monocular depth estimation.

## Overview

The pretraining process involves several steps:

1. **Download pretrained DINOv2 weights**: We start with the weights from the official DINOv2 repository, which have been trained on large-scale image datasets.
2. **Convert weights**: Convert DINOv2 weights to our teacher model format.
3. **Fine-tune on depth data**: Adapt the pretrained model for the specific task of depth estimation.

## Usage

### Automated Pretraining

The easiest way to run the complete pretraining pipeline is to use the provided bash script:

