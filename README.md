# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.  
The project implements an end-to-end pipeline for training an LSTM model and serving predictions through a Flask-based inference API along with a web-based user interface.

---

## Project Status

- ✅ Data loading and preprocessing
- ✅ RUL label generation
- ✅ Feature scaling and sequence creation
- ✅ LSTM model training
- ✅ Model and scaler persistence
- ✅ Inference API for real-time RUL prediction
- ✅ Web UI (frontend) integrated with backend
- ✅ User Authentication System (Sign Up, Login, Logout)
- ✅ SQLite Database for User Management
- ✅ Flask Web Application with UI
- ✅ Password Hashing and Security

---

## Project Structure

```

BeTwin-AI/
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── results/
│   ├── model.h5
│   └── scaler.pkl
├── src/
│   ├── app.py
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── main.py
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── about.html
│   ├── dashboard.html
│   └── auth/
│       ├── login.html
│       └── signup.html
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Dataset

- NASA C-MAPSS turbofan engine dataset
- Includes:
  - Training data
  - Test data
  - True RUL values

---

## LSTM Model

- **Architecture**: LSTM-based regression model
- **Input**: Fixed-length multivariate sensor sequences (30 timesteps × 21 sensors)
- **Output**: Continuous RUL value (Remaining Useful Life in cycles)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

---

## Technology Stack

### Backend

- Python 3.13
- Flask (Web framework)
- TensorFlow / Keras (Deep learning)
- SQLite (Database)
- Werkzeug (Authentication security)

### Frontend

- HTML5 / Jinja2
- Tailwind CSS
- JavaScript

### Data Processing

- NumPy
- Pandas
- Scikit-learn
- Joblib

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (Optional)

```bash
python src/train.py
```

Trained model and scaler will be saved in `results/`.

### 3. Run the Flask Application

```bash
python src/app.py
```

App will run at:

```
http://127.0.0.1:5000
```

---

## Features

### Web Application

- 🏠 Home Page
- 👤 User Authentication (Signup/Login)
- 🔐 Password Hashing & Security
- 📊 Dashboard (Login required)
- ℹ️ About Page

### Authentication API

- **POST /signup**
  - Fields: `fullname`, `email`, `company`, `password`, `confirm_password`

- **POST /login**
  - Fields: `email`, `password`

- **GET /logout**

---

## RUL Prediction API

### POST /predict

#### Request Body

```json
{
  "sensor_data": [[...30 values...], [...30 values...]]
}
```

#### Response

```json
{
  "predicted_RUL": 1.57
}
```

#### Example (PowerShell)

```powershell
$body = @{sensor_data=(1..30|%{,@(0..23|%{0})})} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:5000/predict -Method POST -ContentType application/json -Body $body
```

---

## Database

- SQLite Database: `betwin_ai.db`
- Auto-created on first run

### Users Table

- id (Primary Key)
- fullname
- email (Unique)
- company
- password (hashed)
- created_at

---

## Notes

- API expects exactly 30 timesteps
- Feature count must match training config (21 sensors)
- Models, scalers, and DB are excluded via `.gitignore`
- Passwords are securely hashed
- Uses Flask sessions for authentication
- Database auto-initialized via `init_db()`
- Frontend uses Tailwind CSS with Jinja2 templates

---

## Recent Updates

- Fixed HTML template issues
- Added full authentication system
- Integrated frontend with backend
- Implemented SQLite DB
- Improved project structure
- Application fully functional

---

## Authors

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal

````

---

## ✅ What you should do now

1. Replace your README with this
2. Then run:
```bash
git add README.md
git commit -m "Resolved merge conflict and updated README"
git push
````
