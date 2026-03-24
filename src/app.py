from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
import sqlite3
import os

# Get the parent directory (project root)
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder=os.path.join(PARENT_DIR, 'templates'),
            static_folder=os.path.join(PARENT_DIR, 'static'))
app.secret_key = 'your-secret-key-change-this-in-production'

model = load_model(os.path.join(PARENT_DIR, "results/model.h5"), compile=False)
scaler = joblib.load(os.path.join(PARENT_DIR, "results/scaler.pkl"))
SEQ_LENGTH = 30

# ===== DATABASE INITIALIZATION =====
def init_db():
    conn = sqlite3.connect('betwin_ai.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    fullname TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    company TEXT,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )''')
    conn.commit()
    conn.close()

init_db()
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    return render_template("dashboard.html")

# ===== AUTHENTICATION ROUTES =====
@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        conn = sqlite3.connect('betwin_ai.db')
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = c.fetchone()
        conn.close()
        
        if user and check_password_hash(user[4], password):
            session['user_id'] = user[0]
            session['fullname'] = user[1]
            session['email'] = user[2]
            flash(f'Welcome back, {user[1]}!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password', 'danger')
    
    return render_template("auth/login.html")

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        email = request.form.get('email')
        company = request.form.get('company')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validation
        if not all([fullname, email, password, confirm_password]):
            flash('All fields are required', 'danger')
            return redirect(url_for('signup'))
        
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return redirect(url_for('signup'))
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return redirect(url_for('signup'))
        
        try:
            conn = sqlite3.connect('betwin_ai.db')
            c = conn.cursor()
            hashed_password = generate_password_hash(password)
            c.execute('''INSERT INTO users (fullname, email, company, password)
                        VALUES (?, ?, ?, ?)''',
                     (fullname, email, company, hashed_password))
            conn.commit()
            conn.close()
            
            flash('Account created successfully! Please login.', 'success')
            return redirect(url_for('login'))
        
        except sqlite3.IntegrityError:
            flash('Email already registered. Try logging in instead.', 'warning')
            return redirect(url_for('login'))
    
    return render_template("auth/signup.html")

@app.route("/logout")
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))
@app.route("/predict",methods=["GET","POST"])
def predict():
  if request.method=="POST":
    body=request.get_json()
  else:
    data=request.args.get("sensor_data")
    if data is None:
      return jsonify({"error":"sensor_data query parameter is required"}),400
    try:
      body={"sensor_data":json.loads(data)}
    except Exception:
      return jsonify({"error":"sensor_data must be valid JSON"}),400
  if body is None or "sensor_data" not in body:
    return jsonify({"error":"sensor_data is required"}),400
  sensor_array=np.array(body["sensor_data"])
  if sensor_array.ndim!=2:
    return jsonify({"error":"sensor_data must be a 2D array"}),400
  if sensor_array.shape[0]!=SEQ_LENGTH:
    return jsonify({"error":f"Expected {SEQ_LENGTH} rows, got {sensor_array.shape[0]}"}),400
  if sensor_array.shape[1]!=scaler.n_features_in_:
    return jsonify({"error":f"Expected {scaler.n_features_in_} features, got {sensor_array.shape[1]}"}),400
  scaled_data=scaler.transform(sensor_array)
  model_input=np.expand_dims(scaled_data,axis=0)
  output=model.predict(model_input,verbose=0)
  predicted_rul=float(output[0][0])
  return jsonify({"predicted_RUL":round(predicted_rul,2)})
if __name__=="__main__":
  app.run(debug=True)