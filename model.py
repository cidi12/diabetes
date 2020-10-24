{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import pylab as pl\n",
    "import numpy as np\n",
    "import scipy.optimize as opt\n",
    "from sklearn import preprocessing\n",
    "%matplotlib inline \n",
    "import matplotlib.pyplot as plt\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv ('diabetes.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pregnancies                 111\n",
      "Glucose                       5\n",
      "BloodPressure                35\n",
      "SkinThickness               227\n",
      "Insulin                     374\n",
      "BMI                          11\n",
      "DiabetesPedigreeFunction      0\n",
      "Age                           0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df = pd.read_csv (\"diabetes.csv\")\n",
    "print((df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] == 0).sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pregnancies                   0\n",
      "Glucose                       5\n",
      "BloodPressure                35\n",
      "SkinThickness               227\n",
      "Insulin                     374\n",
      "BMI                          11\n",
      "DiabetesPedigreeFunction      0\n",
      "Age                           0\n",
      "Outcome                       0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)\n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Insulin'] = np.where(df['Insulin'].isna(), \n",
    "                       np.random.uniform(1, 846, size=len(df)), \n",
    "                       df['Insulin'])\n",
    "df['Glucose'] = np.where(df['Glucose'].isna(), \n",
    "                       np.random.uniform(1, 199, size=len(df)), \n",
    "                       df['Glucose'])\n",
    "df['BloodPressure'] = np.where(df['BloodPressure'].isna(), \n",
    "                       np.random.uniform(30, 120, size=len(df)), \n",
    "                       df['BloodPressure'])\n",
    "df['SkinThickness'] = np.where(df['SkinThickness'].isna(), \n",
    "                       np.random.uniform(30, 102, size=len(df)), \n",
    "                       df['SkinThickness'])\n",
    "df['BMI'] = np.where(df['BMI'].isna(), \n",
    "                       np.random.uniform(25.1, 67.1, size=len(df)), \n",
    "                       df['BMI'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[6.00000000e+00, 1.48000000e+02, 7.20000000e+01, 3.50000000e+01,\n",
       "        1.81395034e+02, 3.36000000e+01, 6.27000000e-01, 5.00000000e+01],\n",
       "       [1.00000000e+00, 8.50000000e+01, 6.60000000e+01, 2.90000000e+01,\n",
       "        2.12168749e+00, 2.66000000e+01, 3.51000000e-01, 3.10000000e+01],\n",
       "       [8.00000000e+00, 1.83000000e+02, 6.40000000e+01, 8.58208967e+01,\n",
       "        6.46880470e+02, 2.33000000e+01, 6.72000000e-01, 3.20000000e+01],\n",
       "       [1.00000000e+00, 8.90000000e+01, 6.60000000e+01, 2.30000000e+01,\n",
       "        9.40000000e+01, 2.81000000e+01, 1.67000000e-01, 2.10000000e+01],\n",
       "       [0.00000000e+00, 1.37000000e+02, 4.00000000e+01, 3.50000000e+01,\n",
       "        1.68000000e+02, 4.31000000e+01, 2.28800000e+00, 3.30000000e+01]])"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X = np.asarray(df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']])\n",
    "X[0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 0, 1, 0, 1], dtype=int64)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y = np.asarray(df['Outcome'])\n",
    "y [0:5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import preprocessing\n",
    "X = preprocessing.StandardScaler().fit(X).transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.20, random_state=12)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix\n",
    "logr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
    "          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,\n",
    "          penalty='l1', random_state=None, solver='liblinear', tol=0.0001,\n",
    "          verbose=0, warm_start=False).fit(X_train,y_train)\n",
    "pickle.dump(logr,open('model.pkl','wb'))\n",
    "model=pickle.load(open('model.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 0 1 0 1 1 0 1 0 0 0\n",
      " 1 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0\n",
      " 0 1 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 1 0 0 0 0 1 0 1\n",
      " 1 0 0 1 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 1\n",
      " 0 1 0 1 0 0]\n"
     ]
    }
   ],
   "source": [
    "pred_result = logr.predict(X_test)\n",
    "print(pred_result)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.80      0.91      0.85        99\n",
      "           1       0.79      0.60      0.68        55\n",
      "\n",
      "    accuracy                           0.80       154\n",
      "   macro avg       0.79      0.75      0.77       154\n",
      "weighted avg       0.80      0.80      0.79       154\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(y_test, pred_result))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[40 15]\n",
      " [32 67]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import itertools\n",
    "def plot_confusion_matrix(cm, classes,\n",
    "                          normalize=False,\n",
    "                          title='Confusion matrix',\n",
    "                          cmap=plt.cm.Blues):\n",
    "    \"\"\"\n",
    "    This function prints and plots the confusion matrix.\n",
    "    Normalization can be applied by setting `normalize=True`.\n",
    "    \"\"\"\n",
    "    if normalize:\n",
    "        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]\n",
    "        print(\"Normalized confusion matrix\")\n",
    "    else:\n",
    "        print('Confusion matrix, without normalization')\n",
    "\n",
    "    print(cm)\n",
    "\n",
    "    plt.imshow(cm, interpolation='nearest', cmap=cmap)\n",
    "    plt.title(title)\n",
    "    plt.colorbar()\n",
    "    tick_marks = np.arange(len(classes))\n",
    "    plt.xticks(tick_marks, classes, rotation=45)\n",
    "    plt.yticks(tick_marks, classes)\n",
    "\n",
    "    fmt = '.2f' if normalize else 'd'\n",
    "    thresh = cm.max() / 2.\n",
    "    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):\n",
    "        plt.text(j, i, format(cm[i, j], fmt),\n",
    "                 horizontalalignment=\"center\",\n",
    "                 color=\"white\" if cm[i, j] > thresh else \"black\")\n",
    "\n",
    "    plt.tight_layout()\n",
    "    plt.ylabel('True label')\n",
    "    plt.xlabel('Predicted label')\n",
    "print(confusion_matrix(y_test, pred_result, labels=[1,0]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Confusion matrix, without normalization\n",
      "[[33 22]\n",
      " [ 9 90]]\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Compute confusion matrix\n",
    "cnf_matrix = confusion_matrix(y_test, pred_result, labels=[1,0])\n",
    "np.set_printoptions(precision=2)\n",
    "\n",
    "\n",
    "# Plot non-normalized confusion matrix\n",
    "plt.figure()\n",
    "plot_confusion_matrix(cnf_matrix, classes=['outcome=1','outcome=0'],normalize= False,  title='Confusion matrix')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Jaccardresult 0.515625\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import jaccard_score\n",
    "from sklearn.metrics import log_loss\n",
    "\n",
    "print('Jaccardresult', jaccard_score(y_test, pred_result))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
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
