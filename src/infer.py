import joblib
def predict(text):
    m = joblib.load('./models/task5_tagger.joblib')
    tf = m['tf']; clf = m['clf']
    Xv = tf.transform([text])
    preds = clf.predict_proba(Xv)
    labels = ['login','billing']
    out = {labels[i]: float(preds[i][0][1]) for i in range(len(labels))}
    return out
if __name__=='__main__':
    print(predict('I have a problem with my invoice'))
