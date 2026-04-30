from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import os

from flask import session, redirect, url_for
# ── Flask + DB imports ───────────────────────────────
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt

# ── Paths ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

MODEL_PATH = os.path.join(ROOT_DIR, "results", "model.h5")
SCALER_PATH = os.path.join(ROOT_DIR, "results", "scaler.pkl")
DATA_PATH = os.path.join(ROOT_DIR, "data", "train_FD001.txt")

SEQ_LENGTH = 30

# ── Flask App ─────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(ROOT_DIR, "templates"),
    static_folder=os.path.join(ROOT_DIR, "static")
)

app.secret_key = "betwin_ai_super_secret_key_change_this"
# ── DB CONFIG ─────────────────────────────────────────
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///betwin_ai.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

# ── USER MODEL ────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    company = db.Column(db.String(100))
    password = db.Column(db.String(200), nullable=False)


# create DB (run once)
with app.app_context():
    db.create_all()

# ── Load Data ─────────────────────────────────────────
raw_df = pd.read_csv(DATA_PATH, sep=" ", header=None)
raw_df = raw_df.dropna(axis=1)

max_cycle = raw_df.groupby(0)[1].max().reset_index()
max_cycle.columns = [0, "max_cycle"]
raw_df = raw_df.merge(max_cycle, on=0)
raw_df["RUL"] = raw_df["max_cycle"] - raw_df[1]
raw_df.drop("max_cycle", axis=1, inplace=True)

SENSOR_COLS = raw_df.columns[2:-1].tolist()

# scaler
scaler = joblib.load(SCALER_PATH)
raw_df[SENSOR_COLS] = scaler.transform(raw_df[SENSOR_COLS])

# model
model = load_model(MODEL_PATH, compile=False)

print("[INFO] Data shape:", raw_df.shape)
print("[INFO] Sensors:", len(SENSOR_COLS))


# ── Sequence builder ──────────────────────────────────
def get_engine_sequence(engine_id):
    df = raw_df[raw_df[0] == engine_id].copy()

    if df.empty:
        return None, None

    df = df.sort_values(by=1)

    total_cycles = int(df[1].max())

    if len(df) < SEQ_LENGTH:
        return None, total_cycles

    max_start = len(df) - SEQ_LENGTH
    start_idx = np.random.randint(0, max_start + 1)

    seq = df.iloc[start_idx:start_idx + SEQ_LENGTH]

    X = seq[SENSOR_COLS].values.astype(np.float32)

    return X, total_cycles


# ── Pages ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    return render_template("dashboard.html", name=session.get("user_name"))


# ── LOGIN ─────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        user = User.query.filter_by(email=email).first()

        if not user:
            return render_template("auth/login.html", error="User not found")

        if not bcrypt.check_password_hash(user.password, password):
            return render_template("auth/login.html", error="Incorrect password")

        # 🔐 CREATE SESSION
        session["user_id"] = user.id
        session["user_name"] = user.fullname

        return redirect(url_for("dashboard"))

    return render_template("auth/login.html")


# ── SIGNUP ────────────────────────────────────────────
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        fullname = request.form.get("fullname")
        email = request.form.get("email")
        company = request.form.get("company")
        password = request.form.get("password")

        # check duplicate
        if User.query.filter_by(email=email).first():
            return "User already exists"

        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

        new_user = User(
            fullname=fullname,
            email=email,
            company=company,
            password=hashed_pw
        )

        db.session.add(new_user)
        db.session.commit()

        return "Signup successful"

    return render_template("auth/signup.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/profile")
def profile():
    if "user_id" not in session:
        return redirect(url_for("login"))

    user = User.query.get(session["user_id"])

    return render_template("profile.html", user=user)


# ── PREDICT ───────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json()

    if not body or "engine_id" not in body:
        return jsonify({"error": "engine_id required"}), 400

    engine_id = int(body["engine_id"])

    X, total_cycles = get_engine_sequence(engine_id)

    if X is None:
        return jsonify({"error": "Engine not found or insufficient data"}), 404

    model_input = np.expand_dims(X, axis=0)

    print(f"\nENGINE {engine_id}")
    print("shape:", model_input.shape)
    print("std:", np.std(X))

    pred = float(model.predict(model_input, verbose=0)[0][0])
    pred = max(0.0, pred)

    if pred <= 30:
        health = "critical"
        msg = "Immediate maintenance required"
    elif pred <= 60:
        health = "warning"
        msg = "Schedule maintenance soon"
    else:
        health = "safe"
        msg = "Engine is healthy"

    return jsonify({
        "engine_id": engine_id,
        "predicted_RUL": round(pred, 2),
        "total_cycles": total_cycles,
        "health": health,
        "message": msg
    })


# ── RUN ───────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)