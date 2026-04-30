# 🚀 BeTwin-AI — Predictive Aircraft Engine Health System

BeTwin-AI is a **production-grade, full-stack AI system** designed for **predictive maintenance of aircraft engines** using the NASA C-MAPSS dataset.

It combines:

- 🧠 Deep Learning (LSTM-based time-series forecasting)
- ⚙️ End-to-end ML pipeline (training → validation → inference)
- 🌐 Flask full-stack web application
- 🔐 Enterprise-grade authentication system (Session + JWT)
- 📊 Real-time predictive dashboard
- 🛡️ Secure, deployment-ready architecture

---

# 🎯 Problem Statement

Aircraft engines degrade gradually due to complex operational conditions.

Unexpected failures can cause:

- ❌ High maintenance cost
- ❌ Flight delays
- ❌ Safety risks

### 👉 BeTwin-AI solves this by:

- Predicting **Remaining Useful Life (RUL)**
- Monitoring engine degradation patterns
- Enabling **predictive maintenance before failure**
- Reducing downtime and operational risk

---

# 🧠 System Evolution (REAL-WORLD ML DEBUGGING JOURNEY)

This project was not just built — it was **debugged like a real industrial ML system**.

---

## ❌ Initial Problems

- Same prediction for all engines (≈ 90.24 constant output)
- Broken sliding window logic
- Scaling mismatch between training & inference
- Poor temporal feature learning
- Weak generalization of LSTM model
- Engine ID independence failure

---

## 🔧 FINAL FIXES APPLIED

### 1. 🔄 Correct Sliding Window Engine

- Fixed sequence extraction logic
- Ensured **temporal continuity**
- Removed random inconsistency in inference

---

### 2. 📉 Scaling Consistency Fix

- Unified scaler across training & inference
- Eliminated feature distribution shift
- Fixed silent prediction bias issue

---

### 3. 🧠 LSTM Input Standardization

Enforced strict input shape:

```text
(1, 30, 21)
```

✔ Batch size = 1
✔ Timesteps = 30
✔ Features = 21 sensors

---

### 4. ⚙️ Engine-Specific Prediction Fix

Each engine now generates:

- Unique sensor sequence
- Independent degradation pattern
- Realistic RUL variation

---

### 5. 🔐 Dual Authentication System (ENTERPRISE LEVEL)

BeTwin-AI implements **two-layer security architecture**:

---

## 🔑 SESSION-BASED AUTH (Flask-Login)

- Secure login sessions
- Protected routes (`/dashboard`, `/profile`)
- Auto logout handling
- Password hashing using bcrypt

---

## 🔐 JWT AUTH (API SECURITY LAYER)

Implemented for API endpoints:

- Token-based authentication
- Stateless secure communication
- Ideal for mobile / external integrations

### Example flow:

```text
Login → JWT Token Generated → API Access (/predict)
```

### Protected endpoint:

```python
@jwt_required()
def predict():
```

---

### 🛡️ Security Advantages

- Prevents unauthorized API access
- Protects ML inference endpoint
- Supports scalable microservices integration
- Ready for production deployment

---

# 📊 FINAL SYSTEM ARCHITECTURE

```text
                ┌────────────────────┐
                │ NASA C-MAPSS Data  │
                └─────────┬──────────┘
                          ↓
             ┌─────────────────────────┐
             │ Data Preprocessing      │
             │ - Cleaning              │
             │ - Scaling               │
             │ - RUL Generation        │
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
    │ Output: RUL Regression           │
    └─────────┬────────────────────────┘
              ↓
    ┌──────────────────────────────────┐
    │ Flask Backend API               │
    │ + Session Auth + JWT Security   │
    └─────────┬────────────────────────┘
              ↓
    ┌───────────────────────────┐
    │ Web Dashboard UI          │
    │ (Protected Routes)        │
    └───────────────────────────┘
```

---

# 🔐 SECURITY ARCHITECTURE (HIGHLIGHT)

BeTwin-AI is designed with **multi-layer security principles**:

### 🛡️ 1. Authentication Layer

- Flask-Login (session management)
- Secure password hashing (bcrypt)
- Login-required route protection

### 🔐 2. API Security Layer

- JWT token authentication
- Stateless API access control
- Token-based prediction access

### 🚫 3. Access Control

- Role-based system (admin/user ready)
- Protected dashboard & profile routes
- Unauthorized access blocking

### 🔒 4. Data Security

- SQLite secure local DB
- Hashed credentials storage
- No plaintext password storage

---

# 🧠 MACHINE LEARNING MODEL

## Model Type

- LSTM (Long Short-Term Memory Network)

## Input Shape

```text
30 timesteps × 21 sensor features
```

## Output

```text
Continuous RUL prediction (regression)
```

## Training Strategy

- Sliding window time-series learning
- MSE loss function
- Adam optimizer
- Sequence-based supervised learning

---

# 🔥 KEY FIX: “90.24 CONSTANT PREDICTION BUG”

## ❌ Root Cause

- Random window sampling
- Scaling mismatch
- Temporal inconsistency
- Engine identity leakage

## ✅ Final Fix

- Fixed last-30-cycle window extraction
- Ensured consistent scaler usage
- Preserved time-series order
- Engine-specific inference pipeline

---

# 🌐 FLASK WEB SYSTEM

## 🏠 Pages

- Home
- About
- Dashboard (Protected)
- Profile (Protected)
- Login / Signup

---

## 🔐 Authentication System

- User registration system
- Secure login/logout
- Session persistence
- JWT API authentication
- Route protection middleware

---

# 🔌 API REFERENCE

## 🔮 Predict RUL

### Endpoint

```http
POST /predict
```

### JWT Required

```text
Authorization: Bearer <token>
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

# ⚙️ TECH STACK

## Backend

- Flask
- Flask-Login
- Flask-JWT-Extended
- Flask-Bcrypt
- SQLAlchemy
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

# 🚀 DEPLOYMENT READY FEATURES

✔ Production-ready Flask architecture
✔ JWT-secured APIs
✔ Session authentication system
✔ Scalable ML inference pipeline
✔ Modular project structure
✔ Cloud deployment compatible

---

# 🧪 DEBUGGING ACHIEVEMENTS

- Fixed constant prediction collapse (90.24 issue)
- Fixed scaling inconsistency bug
- Fixed LSTM sequence mismatch
- Fixed authentication bypass vulnerability
- Stabilized ML inference pipeline
- Improved model generalization

---

# 👨‍💻 AUTHORS

- Riddhima Rajput
- Diksha Sharma
- Charvi Mittal

---

# 🚀 FUTURE ROADMAP

- Real-time IoT sensor streaming
- Dockerized microservice deployment
- AWS / Render production scaling
- Model retraining pipeline automation
- Advanced Transformer-based RUL model
- Explainable AI (XAI) for engine failure reasons

---

# ⭐ PROJECT STATUS

✔ Fully functional AI + Web System
✔ Secure authentication (Session + JWT)
✔ Production-grade ML pipeline
✔ Deployment-ready architecture
✔ Fixed major predictive modeling issues

---
