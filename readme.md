
# ISBI_PS3C Champion Solution 🏆

**First Place Solution** using Foundation Models and Ensemble Learning

## 📌 Overview

This repository contains the champion solution for the ISBI_PS3C challenge, leveraging state-of-the-art foundation models combined with ensemble learning strategies. Our approach achieves superior performance through:

- **Low-rank Adaptation**: Low-rank Adaptation for pathology foundation models.
- **Multi-model Ensemble**: Integration of UNI, GigaPath, and Hoptimus.
- **Advanced Framework**: Built on AutoGluon 1.2 for automated machine learning.

## 🚀 Quick Start

### Prerequisites

```bash
pip install autogluon==1.2
```

# Train UNI model

python main.py --encoder uni --save_dir [YOUR_SAVE_PATH] --batch_size 64

# Train GigaPath model

python main.py --encoder gigapath --save_dir [YOUR_SAVE_PATH] --batch_size 32

# Train Hoptimus model

python main.py --encoder hoptimus --save_dir [YOUR_SAVE_PATH] --batch_size 32


# Model inference

python infer.py (infer_ISBI_test)

# Train the ensemble model:

python autogluon_table.py

# Ensemble model inference

python infer.py (autogluon_infer)
