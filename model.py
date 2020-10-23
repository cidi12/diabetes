{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[1.00e+00 1.00e+02 6.60e+01 ... 3.20e+01 4.44e-01 4.20e+01]\n",
      " [9.00e+00 5.70e+01 8.00e+01 ... 3.28e+01 9.60e-02 4.10e+01]\n",
      " [0.00e+00 1.00e+02 7.00e+01 ... 3.08e+01 5.97e-01 2.10e+01]\n",
      " ...\n",
      " [3.00e+00 1.30e+02 7.80e+01 ... 2.84e+01 3.23e-01 3.40e+01]\n",
      " [0.00e+00 9.50e+01 6.40e+01 ... 4.46e+01 3.66e-01 2.20e+01]\n",
      " [3.00e+00 1.16e+02 0.00e+00 ... 2.35e+01 1.87e-01 2.30e+01]]\n",
      "[0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1\n",
      " 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0\n",
      " 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0\n",
      " 0 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 1 1 0 0 0 1]\n",
      "[0 0 0 0 0 1 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 1 0 0 1 0 0 0 0 0 1\n",
      " 0 0 1 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0\n",
      " 1 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 1 0\n",
      " 0 1 1 0 0 0 0 0 0 1 1 0 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 1 1 0 0 0 1]\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask\n",
    "from flask import render_template,redirect,request\n",
    "import pandas as pd\n",
    "import sys\n",
    "import numpy as np\n",
    "import pickle\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn import preprocessing\n",
    "\n",
    "#from sklearn.metrics import mean_squared_log_error\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "\n",
    "df=pd.read_csv('diabetes.csv')\n",
    "X = np.asarray(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])\n",
    "X[0:5]\n",
    "y = np.asarray(df['Outcome'])\n",
    "y [0:5]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=4)\n",
    "X = preprocessing.StandardScaler().fit(X).transform(X)\n",
    "X[0:5]\n",
    "\n",
    "logr=LogisticRegression(C=0.01, solver='liblinear')\n",
    "logr.fit(X_train,y_train)\n",
    "#new_data=np.array(new_data,dtype='int64')\n",
    "#new_data=new_data.reshape(1,13)\n",
    "#xnew_data=pd.DataFrame(new_data)\n",
    "pickle.dump(logr,open('model.pkl','wb'))\n",
    "model=pickle.load(open('model.pkl','rb'))\n",
    "result=model.predict(X_test)\n",
    "newdata=X_test\n",
    "print(newdata)\n",
    "result2=model.predict(newdata)\n",
    "print(result)\n",
    "print(result2)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
