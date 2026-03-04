# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.  
The project implements an end-to-end pipeline for training an LSTM model and serving predictions through a lightweight Flask inference API, along with a web-based UI (frontend).

## Project Status

- Data loading and preprocessing
- RUL label generation
- Feature scaling and sequence creation
- LSTM model training
- Model and scaler persistence
- Inference API for real-time RUL prediction
- Web-based UI (frontend) for interacting with the system

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
├── templates/  
│ ├── home.html  
│ ├── about.html  
│ └── dashboard.html  
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

The trained model and scaler are saved in the `results/` directory.

## Run the Inference API and UI

Start the application (from project root)

python src/app.py

The service runs at

http://127.0.0.1:5000

## Web UI (Frontend)

The project also includes a simple frontend built using Flask templates.

Available pages:

- Home page

http://127.0.0.1:5000/

- About page

http://127.0.0.1:5000/about

- Dashboard page

http://127.0.0.1:5000/dashboard

The UI provides a basic interface for navigating the project and viewing the prediction workflow.

## Predict RUL (API)

Endpoint

POST /predict

The API also supports GET requests using a query parameter.

Expected input format

A 2D array with:

- 30 time steps
- N features (must match the trained model configuration)

Example request body

{
"sensor_data": [[...], [...], ...]
}

Example (PowerShell)

$body=@{sensor_data=(1..30|%{,@(0..23|%{0})})}|ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:5000/predict
-Method POST -ContentType application/json -Body $body

Example response

{
"predicted_RUL": 1.57
}

## Notes

- The API expects exactly 30 time steps per request.
- The number of features must exactly match the features used during training.
- The `results/` directory (trained model and scaler) is excluded from version control using `.gitignore`.
- To run the API and UI on a fresh machine, the model must be generated first using `python src/train.py`.

## Authors

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal
