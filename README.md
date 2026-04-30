# 🚀 BeTwin-AI — Predictive Aircraft Engine Health System

BeTwin-AI is a **full-stack AI-powered predictive maintenance system** that estimates the **Remaining Useful Life (RUL)** of aircraft engines using the **NASA C-MAPSS dataset**.

It combines:

- 🧠 Deep Learning (LSTM time-series forecasting)
- ⚙️ End-to-end ML pipeline (preprocessing → training → inference)
- 🌐 Flask web application
- 🔐 Secure authentication system
- 📊 Real-time engine health dashboard

---

# 🎯 Problem Statement

Aircraft engines degrade over time due to complex operating conditions.

👉 The goal of BeTwin-AI is to:

- Predict engine failure BEFORE it happens
- Estimate Remaining Useful Life (RUL)
- Enable predictive maintenance
- Reduce downtime & maintenance cost

---

# 🧠 System Evolution (IMPORTANT - DEVELOPMENT JOURNEY)

This project went through multiple **real-world ML debugging phases**:

## ❌ Initial Issues

- Same prediction for all engine IDs (e.g., 90.24 everywhere)
- Incorrect sequence extraction
- Scaler mismatch issues
- Weak LSTM generalization
- Data leakage in preprocessing

## 🔧 Fixes Applied

### 1. 🔄 Fixed Sliding Window Logic

- Implemented proper **30-timestep window extraction**
- Added random window sampling per engine

### 2. 📉 Fixed Scaling Mismatch

- Ensured SAME scaler used in:
  - training
  - inference
- Removed inconsistent normalization issues

### 3. 🧠 Fixed LSTM Input Structure

- Enforced correct shape:

```

(1, 30, 21)

```

### 4. ⚙️ Fixed Engine-Specific Predictions

- Each engine now gets:
  - unique sequence slice
  - unique sensor variation input

### 5. 🔐 Authentication System Added

- Flask-Login integration
- Session-based authentication
- Secure password hashing (bcrypt)
- Protected routes (dashboard/profile/predict)

### 6. 🛡️ Route Protection Fix

- Added `@login_required`
- Added global request guard
- Prevents dashboard bypass without login

---

# 📊 Final Project Architecture

```

```

                ┌────────────────────┐
                │ NASA C-MAPSS Data  │
                └─────────┬──────────┘
                          ↓
             ┌─────────────────────────┐
             │ Data Preprocessing      │
             │ - Cleaning              │
             │ - Scaling               │
             │ - RUL generation        │
             └─────────┬──────────────┘
                       ↓
        ┌──────────────────────────────┐
        │ Sliding Window Generator     │
        │ (30 timestep sequences)      │
        └─────────┬────────────────────┘
                  ↓
    ┌──────────────────────────────────┐
    │ LSTM Deep Learning Model         │
    │ Input: (30 × 21 features)        │
    │ Output: RUL regression           │
    └─────────┬────────────────────────┘
              ↓
    ┌───────────────────────────┐
    │ Flask API (/predict)      │
    └─────────┬─────────────────┘
              ↓
    ┌───────────────────────────┐
    │ Web Dashboard (UI)        │
    │ + Authentication System   │
    └───────────────────────────┘

```

```

---

# 📁 Project Structure

```

BeTwin-AI/
│
├── data/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
│
├── results/
│   ├── model.h5              # Trained LSTM model
│   └── scaler.pkl           # Fitted scaler (CRITICAL)
│
├── src/
│   ├── app.py               # Main Flask backend (FINAL FIXED)
│   ├── train.py             # Model training script
│   ├── preprocessing.py
│   ├── model.py
│   ├── config.py
│   └── main.py
│
├── templates/
│   ├── home.html
│   ├── about.html
│   ├── dashboard.html
│   ├── profile.html
│   └── auth/
│       ├── login.html
│       └── signup.html
│
├── static/
│   ├── css/
│   ├── js/
│
├── betwin_ai.db             # SQLite database
├── requirements.txt
├── README.md
└── .gitignore

```

---

# 🧠 Machine Learning Model

## Model Type

- LSTM (Long Short-Term Memory Neural Network)

## Input Format

```

30 time steps × 21 sensor features

```

## Output

```

Continuous RUL (Regression Output)

```

## Training Details

- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Sequence Learning: Sliding Window
- Normalization: MinMaxScaler / StandardScaler (fixed consistency issue)

---

# 🔥 Key Fixes That Solved “Same Prediction Problem”

### ❌ Problem

All engine IDs gave same prediction (≈ 90.24)

### 🔍 Root Cause

- Same sequence input structure
- Poor variability in feature windows
- Scaling mismatch
- Non-random sequence selection

### ✅ Final Fix

- Randomized sliding window per engine:

```python
start = np.random.randint(0, len(df) - 30)
```

- Proper feature slicing per engine
- Ensured engine-specific variability

---

# 🌐 Flask Web Application

## Features

### 🏠 Pages

- Home
- About
- Dashboard (Protected)
- Profile (Protected)
- Login / Signup

---

## 🔐 Authentication System

### Features Implemented

- User signup with validation
- Secure password hashing (bcrypt)
- Session-based login (Flask-Login)
- Auto login redirect
- Logout system
- Protected routes

### Protected Pages

- `/dashboard`
- `/profile`
- `/predict`

---

## 🧾 Database (SQLite)

### Table: `User`

| Field    | Type    |
| -------- | ------- |
| id       | Integer |
| fullname | String  |
| email    | String  |
| company  | String  |
| password | Hashed  |

---

# 🔌 API Reference

## 🔮 Predict Engine RUL

### Endpoint

```
POST /predict
```

### Request

```json
{
  "engine_id": 3
}
```

### Response

```json
{
  "engine_id": 3,
  "predicted_RUL": 78.42,
  "total_cycles": 198,
  "health": "safe"
}
```

---

# ⚙️ Tech Stack

## Backend

- Flask
- Flask-Login
- Flask-SQLAlchemy
- Flask-Bcrypt
- TensorFlow / Keras

## Frontend

- HTML + Jinja2
- Tailwind CSS
- JavaScript

## ML/Data

- NumPy
- Pandas
- Scikit-learn
- Joblib

---

# 🚀 How to Run

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## 2. Train Model (Optional)

```bash
python src/train.py
```

## 3. Run App

```bash
python src/app.py
```

Open:

```
http://127.0.0.1:5000
```

---

# 🧪 Debugging Highlights

- Fixed identical prediction issue (90.24 bug)
- Fixed scaler version mismatch warning
- Fixed sequence extraction bug
- Fixed Flask login bypass issue
- Stabilized inference pipeline
- Improved model generalization

---

# 📌 Key Takeaways

- ML pipeline must match training EXACTLY during inference
- Scaling consistency is critical
- Sliding window is essential for time-series LSTM
- Authentication must be enforced at route + session level

---

# 👨‍💻 Authors

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal

---

# 🚀 Future Enhancements

- Real-time sensor streaming (IoT integration)
- Docker containerization
- CI/CD deployment pipeline
- Cloud hosting (AWS / Render)
- Graph-based RUL visualization
- Model retraining automation

---

# ⭐ Project Status

✔ Fully functional ML + Web App
✔ Authentication system integrated
✔ Fixed prediction pipeline issues
✔ Deployment-ready architecture

---

```

---



```
