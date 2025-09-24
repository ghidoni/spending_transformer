
# Credit Risk Model with NuFormer

This project implements a credit risk model using a Transformer-based architecture called NuFormer. The model is pre-trained on customer transaction data and then fine-tuned for credit risk classification.

## Setup

1. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Usage

1. **Generate dummy data:**
```bash
python generate_dummy_data.py
```

2. **Pre-process data and train tokenizer:**
```bash
python data_preprocessor.py
```

3. **Pre-train the model:**
```bash
python pretrain.py
```

4. **Fine-tune the model for classification:**
```bash
python finetune.py
```
