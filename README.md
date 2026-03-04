# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.
The project implements an end-to-end pipeline for training an LSTM model and serving predictions through a lightweight Flask inference API.

## Project Status

- Data loading and preprocessing
- RUL label generation
- Feature scaling and sequence creation
- LSTM model training
- Model and scaler persistence
- Inference API for real-time RUL prediction

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
│ ├── train.py
│ └── app.py
├── requirements.txt
├── README.md
└── .gitignore

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001 subset).

## Model

- LSTM-based regression model
- Input: fixed-length multivariate sensor sequences
- Output: continuous RUL value
- Loss: Mean Squared Error
- Optimizer: Adam

## How to Run

Install dependencies

pip install -r requirements.txt

Train the model (from project root)

python src/train.py

The trained model and scaler are saved in the results/ directory.

## Run the Inference API

Start the API server

python src/app.py

The service runs at

http://127.0.0.1:5000

Health check

Open in browser

http://127.0.0.1:5000/

## Predict RUL

Endpoint

POST /predict

Expected JSON body

{
"sensor_data": [30 x N sensor matrix]
}

Example (PowerShell)

$body=@{sensor_data=(1..30|%{,@(0..23|%{0})})}|ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:5000/predict -Method POST -ContentType application/json -Body $body

Response

{
"predicted_RUL": 1.57
}

## Notes

- The API expects exactly 30 time steps per request.
- The number of features must match the training configuration.
- Generated artifacts such as trained models and scalers are excluded from version control using .gitignore.

## Authors

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal
