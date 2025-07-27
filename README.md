# Temperature Prediction with PyTorch

Predict daily temperatures for the next 15 days using LSTM neural networks.

## Features

- Data preprocessing and normalization
- LSTM model implementation
- Training pipeline
- Prediction visualization

## Installation

1. Clone the repository
```bash
git clone https://github.com/tu-usuario/temperature-prediction-pytorch.git
```
2. Create and activate environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
1. Train the model:
```bash
python src/models/train.py
```
2. Make predictions:
```bash
python src/models/predict.py
```
