# When Batch Normalization Breaks Self-Supervised Time Series Fine-Tuning: A Systematic Study of Optimization Collapse

Official implementation accompanying the paper:

**“When Batch Normalization Breaks Self-Supervised Time Series Fine-Tuning: A Systematic Study of Optimization Collapse”**

This repository provides the code used to reproduce the experiments and analyses presented in the paper, including the large-scale benchmark of Batch Normalization (BN) strategies for fine-tuning self-supervised time series classification models.

## Overview

Batch Normalization layers accumulate running statistics during self-supervised pre-training, but their treatment during downstream fine-tuning remains poorly understood in time series learning. This repository implements and evaluates multiple BN adaptation strategies under a controlled experimental setting involving:

* **6 Batch Normalization strategies**
* **4 deep learning architectures**
* **31 UCR/UEA datasets**
* **3 label ratios**
* **3 mini-batch sizes**
* **5 random seeds**

In total, the benchmark comprises **32,085 experimental runs**.

The experiments evaluate whether BN statistics should be preserved, updated, partially adapted, or replaced during fine-tuning after self-supervised pre-training.

## Repository Structure

```text
.
├── bn_exp.py        # Main experimental pipeline
├── sanity.py        # Sanity-check validation experiments
└── README.md
```

## Main Scripts

### `bn_exp.py`

This is the **main experimental pipeline** used in the paper.

It performs the systematic benchmark across BN strategies, architectures, datasets, label ratios, batch sizes, and random seeds. The script handles:

* Self-supervised pre-training configuration
* Fine-tuning with different BN strategies
* Controlled experimental variations
* Metric computation (Macro-F1, accuracy, precision, recall)
* Aggregation of results across runs

This script reproduces the primary findings reported in the manuscript.

### `sanity.py`

This script implements the **sanity-check experiments** described in the paper.

Its purpose is to verify that the poor performance observed for some BN strategies (particularly full BN update under SSL fine-tuning) is not caused by implementation errors. The script compares:

* **Standard supervised training from scratch**
* **SSL fine-tuning using BN update strategies**

The sanity experiments confirm that the observed failure mode is specific to the SSL fine-tuning setting and not an artifact of the implementation.

## Reproducibility

The repository was designed with reproducibility in mind.

All experiments follow the protocol described in the paper, including:

* Fixed random seeds
* Controlled label ratios
* Standardized preprocessing
* Consistent optimization settings
* Architecture-specific configurations

Results may vary slightly depending on hardware, software versions, and parallelization settings.

## Citation

If you use this repository or build upon this work, please cite:
<!--
```bibtex
@article{goyo2026bnstrategies,
  title={Freezing or Adapting? A Systematic Benchmark of Batch Normalization Strategies for Fine-Tuning Self-Supervised Time Series Classifiers},
  author={Goyo, Manuel Alejandro and Icarte-Ahumada, Gabriel and Contreras, German and Hidalgo, Mauricio},
  year={2026}
}
```
-->
## Contact

For questions, issues, or research collaborations, please open an issue or contact the authors through the information provided in the paper.
