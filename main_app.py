from flask import Flask
from flask import render_template,redirect,request
import pandas as pd
import sys
import numpy as np
import pickle
app=Flask(__name__)
    
model=pickle.load(open('model.pkl','rb'))
@app.route("/")
def index():
    return render_template("index.html")
    
@app.route("/result", methods = ['POST', 'GET'])
def result():
        request.method=='POST'
        Pregnancies = request.form["Pregnancies"]
        Glucose=request.form["Glucose"]
        BloodPressure=request.form["BloodPressure"]	
        SkinThickness=request.form["SkinThickness"]
        Insulin=request.form["Insulin"]
        BMI=request.form["BMI"]
        DiabetesPedigreeFunction=request.form["DiabetesPedigreeFunction"]
        Age=request.form["Age"]
        lst=list()
        lst.append((Pregnancies))
        lst.append((Glucose))
        lst.append((BloodPressure))
        lst.append((SkinThickness))
        lst.append((Insulin))
        lst.append((BMI))
        lst.append((DiabetesPedigreeFunction))
        lst.append((Age))
        ans=model.predict([np.array(lst,dtype='float')])
        result=ans
        print(ans)
        print(lst)
        
        return render_template("index.html",result=result,lst=lst)
        

if __name__ == "__main__":
    app.run(port=5000,debug=True)