from flask import Flask,request,jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import json
app=Flask(__name__)
model=load_model("../results/model.h5",compile=False)
scaler=joblib.load("../results/scaler.pkl")
SEQ_LENGTH=30
@app.route("/")
def home():
  return "RUL Prediction API is running"
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