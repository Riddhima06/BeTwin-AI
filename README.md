# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the **Remaining Useful Life (RUL)** of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.  
The current version implements a complete end-to-end **training pipeline** using an LSTM-based model.

## Project Status

- Implemented data loading and preprocessing
- RUL label generation
- Feature scaling and sequence creation
- LSTM model training
- Model and scaler persistence

Future updates will include evaluation metrics, inference pipeline, visualization, and support for additional datasets.

## Project Structure

BeTwin-AI/
├── data/
│ ├── train_FD001.txt
│ ├── test_FD001.txt
│ └── RUL_FD001.txt
├── results/
├── src/
│ ├── config.py
│ ├── preprocessing.py
│ ├── model.py
│ └── train.py
├── requirements.txt
├── README.md
└── .gitignore

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001).

## Model

- LSTM-based regression model
- Input: Fixed-length sensor sequences
- Output: Continuous RUL prediction
- Loss: Mean Squared Error
- Optimizer: Adam

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt

Run training from project root:

python src/train.py

Trained artifacts are saved in the results/ directory.

Notes

Generated files such as trained models and scalers are excluded from version control.

Authors:
Riddhima Rajput
Diksha Sharma
Charvi Mittal
```
