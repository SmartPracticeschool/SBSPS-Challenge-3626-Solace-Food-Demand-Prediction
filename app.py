from flask import Flask, render_template, request, redirect, url_for, send_file, make_response,jsonify
import pandas as pd
import pickle
import xgboost as xgb
from helpers import *
import json
#with open("models/model","rb") as file:

bst = xgb.Booster({'nthread': 4})  # init model
bst.load_model("models/model")
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]="./"
@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html",context={"name": "Forecast"})
@app.route("/predict_csv",methods=["POST"])
def predict_csv_file():
    if request.method=="POST":
        print(request.files)
        file=request.files["csv_file"]
        predictions = predict_csv(file,bst)
        #return send_file(predcsv)
        resp = make_response(predictions.to_csv(index=False))
        resp.headers["Content-Disposition"] = "attachment; filename=export.csv"
        resp.headers["Content-Type"] = "text/csv"
        return resp



@app.route("/predict",methods=["POST"])
def predict():
    data=request.form.to_dict()
    print(data)
    preds=predict_individual(data,bst)
    print(preds)
    return render_template("prediction.html",context={"data":data,"pred":preds[0][0]})


if __name__ == '__main__':
    app.run(port=8100, debug=True)

