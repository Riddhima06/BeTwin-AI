# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.  
The project implements an end-to-end pipeline for training an LSTM model and serving predictions through a Flask-based inference API along with a web-based user interface.

---

## Project Status

- Data loading and preprocessing
- RUL label generation
- Feature scaling and sequence creation
- LSTM model training
- Model and scaler persistence
- Inference API for real-time RUL prediction
- Web UI (frontend) integrated with the backend for live predictions

---

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

---

## Dataset

NASA C-MAPSS Turbofan Engine Degradation Dataset (FD001 subset).

---

## Model

- LSTM-based regression model
- Input: fixed-length multivariate sensor sequences
- Output: continuous RUL value
- Loss: Mean Squared Error
- Optimizer: Adam

---

## How to Run

Install dependencies

pip install -r requirements.txt

Train the model (from project root)

python src/train.py

The trained model and scaler are saved in the `results/` directory.

---

## Run the Web Application and Inference API

The backend API and the frontend UI are served from a single Flask application.

Start the server from the project root:

python src/app.py

The application runs at:

http://127.0.0.1:5000

---

## Frontend (UI)

A web-based user interface has been added to the project.

Pages available:

- Home page
- About page
- Dashboard page

The Dashboard provides a simple interface to trigger the prediction and display the Remaining Useful Life (RUL) returned by the trained model.

When the **LAUNCH MONITOR** button is clicked on the dashboard, the frontend sends sensor data to the backend `/predict` API and displays the predicted RUL on the screen.

---

## Predict RUL (API)

Endpoint

POST /predict

Expected JSON body

{
"sensor_data": [30 x N sensor matrix]
}

Example (PowerShell)

$body=@{sensor_data=(1..30|%{,@(0..23|%{0})})}|ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:5000/predict
-Method POST -ContentType application/json -Body $body

Example response

{
"predicted_RUL": 1.57
}

---

## Notes

- The API expects exactly 30 time steps per request.
- The number of features must match the model training configuration.
- The frontend dashboard internally calls the same `/predict` endpoint.
- Generated artifacts such as trained models and scalers are excluded from version control using `.gitignore`.
- The main application entry point is `src/app.py`. The old root-level `app.py` has been removed and all routes are merged into the single backend.

---

## Authors

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal
