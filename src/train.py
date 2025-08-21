import os
os.makedirs('./models', exist_ok=True)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib

def main():
    # sample dataset
    df = pd.DataFrame({'text':['I cannot login','Billing issue with invoice'],'login':[1,0],'billing':[0,1]})
    X = df['text']; y = df[['login','billing']]
    tf = TfidfVectorizer(max_features=2000)
    Xv = tf.fit_transform(X)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
    clf.fit(Xv, y)
    joblib.dump({'tf':tf,'clf':clf}, './models/task5_tagger.joblib')

if __name__=='__main__':
    main()
