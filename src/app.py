import os
import json
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
import joblib
from tensorflow.keras.models import load_model

BASE_DIR=os.path.dirname(os.path.abspath(__file__))

MODEL_PATH=os.path.join(BASE_DIR,"..","results","model.h5")
SCALER_PATH=os.path.join(BASE_DIR,"..","results","scaler.pkl")
DATA_PATH=os.path.join(BASE_DIR,"..","data","train_FD001.txt")

app=Flask(__name__,template_folder="../templates")

model=load_model(MODEL_PATH,compile=False)
scaler=joblib.load(SCALER_PATH)

# load dataset once (real engine data)
raw_df=pd.read_csv(DATA_PATH,sep=" ",header=None)
raw_df=raw_df.dropna(axis=1)

SEQ_LENGTH=30

# you can tune this later
SAFE_RUL_THRESHOLD=90


@app.route("/")
def home():
  return render_template("home.html")


@app.route("/about")
def about():
  return render_template("about.html")


@app.route("/dashboard")
def dashboard():
  return render_template("dashboard.html")


@app.route("/predict",methods=["POST"])
def predict():

  body=request.get_json()

  if body is None:
    return jsonify({"error":"JSON body is required"}),400

  engine_id=body.get("engine_id")

  if engine_id is None:
    return jsonify({"error":"engine_id is required"}),400

  try:
    engine_id=int(engine_id)
  except:
    return jsonify({"error":"engine_id must be integer"}),400

  # filter that engine
  engine_df=raw_df[raw_df[0]==engine_id]

  if len(engine_df)==0:
    return jsonify({"error":"engine not found"}),400

  if len(engine_df)<SEQ_LENGTH:
    return jsonify({"error":"not enough cycles for this engine"}),400

  # take last 30 cycles
  last_cycles=engine_df.tail(SEQ_LENGTH)

  # take only sensor columns
  sensor_array=last_cycles.iloc[:,2:].values

  # safety check for feature count
  if sensor_array.shape[1]!=scaler.n_features_in_:
    return jsonify({
      "error":f"model expects {scaler.n_features_in_} features, got {sensor_array.shape[1]}"
    }),400

  scaled_data=scaler.transform(sensor_array)

  model_input=np.expand_dims(scaled_data,axis=0)

  output=model.predict(model_input,verbose=0)

  predicted_rul=float(output[0][0])

  # simple health logic
  if predicted_rul <= SAFE_RUL_THRESHOLD:
    health="danger"
    message="⚠ Engine health is critical"
  else:
    health="safe"
    message="✔ Engine health is within safe limits"

  return jsonify({
    #"engine_id":engine_id,
    "predicted_RUL":round(predicted_rul,2),
    "health":health,
    "message":message
  })


if __name__=="__main__":
  app.run(debug=True)