# BeTwin-AI

BeTwin-AI is a deep learning project for predicting the Remaining Useful Life (RUL) of aircraft engines using multivariate time-series sensor data from the NASA C-MAPSS dataset.
The project implements an end-to-end pipeline for training an LSTM model and serving predictions through a lightweight Flask inference API.

## Project Status

- ✅ Data loading and preprocessing
- ✅ RUL label generation
- ✅ Feature scaling and sequence creation
- ✅ LSTM model training
- ✅ Model and scaler persistence
- ✅ Inference API for real-time RUL prediction
- ✅ User Authentication System (Sign Up, Login, Logout)
- ✅ SQLite Database for User Management
- ✅ Flask Web Application with UI
- ✅ Password Hashing and Security

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
│   ├── app.py                (Flask backend with auth & predictions)
│   ├── config.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── main.py
├── templates/
│   ├── base.html             (Base template)
│   ├── home.html             (Homepage)
│   ├── about.html
│   ├── dashboard.html
│   └── auth/
│       ├── login.html        (Login page)
│       └── signup.html       (Registration page)
├── requirements.txt
├── README.md
├── .gitignore
└── activate.bat
```

## LSTM Model

- **Architecture**: LSTM-based regression model
- **Input**: Fixed-length multivariate sensor sequences (30 timesteps × 21 sensors)
- **Output**: Continuous RUL value (Remaining Useful Life in cycles)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam
- **Training Data**: NASA C-MAPSS turbofan degradation dataset

## Technology Stack

### Backend
- **Python 3.13**
- **Flask** - Web framework
- **TensorFlow/Keras** - Deep learning for LSTM model
- **SQLite** - User database
- **Werkzeug** - Password hashing and security

### Frontend
- **HTML5/Jinja2** - Template engine
- **Tailwind CSS** - Responsive styling
- **JavaScript** - Client-side interactions

### Data Processing
- **NumPy** - Numerical operations
- **Pandas** - Data manipulation
- **Scikit-learn** - Preprocessing and scaling
- **Joblib** - Model persistence

## How to Run

### Prerequisites
Install dependencies

```bash
pip install -r requirements.txt
```

### Train the Model (Optional)
From project root:

```bash
python src/train.py
```

The trained model and scaler are saved in the `results/` directory.

### Run the Flask Application

Start the Flask server:

```bash
python src/app.py
```

The application runs at:

```
http://127.0.0.1:5000
```

## Features

### Web Application
- **🏠 Home Page**: Beautiful landing page with project information
- **👤 User Authentication**: Secure sign-up and login system
- **🔐 Password Security**: Passwords are hashed using werkzeug security
- **📊 Dashboard**: User dashboard (requires login)
- **ℹ️ About Page**: Project information and features

### Authentication API
- **POST /signup**: Create a new user account
  - Required fields: `fullname`, `email`, `company`, `password`, `confirm_password`
  - Validation: Email uniqueness, password length (min 8 chars), password match

- **POST /login**: Login to existing account
  - Required fields: `email`, `password`
  - Returns: Session with user_id, fullname, email

- **GET /logout**: Clear user session and logout

### ML Prediction API
- **POST /predict**: Get RUL prediction from sensor data

## RUL Prediction Endpoint

### POST /predict

Expected JSON body:

```json
{
  "sensor_data": [[...30 values...], [...30 values...], ... [30 rows total]]
}
```

Response:

```json
{
  "predicted_RUL": 1.57
}
```

### Example (PowerShell)

```powershell
$body = @{sensor_data=(1..30|%{,@(0..23|%{0})})} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:5000/predict -Method POST -ContentType application/json -Body $body
```

## Database

- **SQLite Database**: `betwin_ai.db`
- **Users Table**: Stores user credentials and profile information
  - id (INTEGER PRIMARY KEY)
  - fullname (TEXT)
  - email (TEXT UNIQUE)
  - company (TEXT)
  - password (TEXT - hashed)
  - created_at (TIMESTAMP)

**Note**: `betwin_ai.db` is automatically created on first run and should NOT be pushed to GitHub.

## Recent Updates (March 24, 2026)

- ✅ Fixed HTML template syntax errors in `base.html` and `home.html`
- ✅ Consolidated authentication code to `src/app.py`
- ✅ Implemented full user authentication system (Sign Up, Login, Logout)
- ✅ Added password hashing and security validation
- ✅ Created SQLite database schema for users
- ✅ Set up Flask with proper template and static file paths
- ✅ Installed all required dependencies successfully
- ✅ Application running successfully on http://localhost:5000

## Completed Tasks

1. **Backend Authentication**
   - User registration with validation
   - Secure password hashing
   - Login with session management
   - Logout functionality

2. **Frontend**
   - Fixed Jinja2 template variables in Tailwind CSS
   - Created responsive authentication pages
   - Dashboard with login protection

3. **Database**
   - SQLite user database setup
   - Automatic schema creation on first run

4. **Deployment Ready**
   - All dependencies installed
   - Application fully functional
   - Code consolidated and optimized

---

- The API expects exactly 30 time steps per request.
- The number of features must match the training configuration (21 sensors).
- Generated artifacts (models, scalers, databases) are excluded from version control.
- All sensitive data (passwords) is securely hashed using werkzeug
- The application uses Flask sessions for user authentication
- Database is created automatically on first run via `init_db()` function
- HTML templates use Jinja2 templating with responsive design
- Frontend uses Tailwind CSS for styling with custom animations

## Authors

Riddhima Rajput
Diksha Sharma
Charvi Mittal
