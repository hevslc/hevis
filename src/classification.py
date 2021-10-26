from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import streamlit as st

@st.cache
def runSVC(corpus, target):
    with st.spinner('Running Classification...'):		
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(corpus, target, test_size=0.30, random_state=42)
        clf = svm.SVC(kernel='rbf', gamma=0.0005, random_state=42) 
        clf.fit(Xtrain, Ytrain)
        Ypred = clf.predict(Xtest)
    
    return f1_score(Ytest, Ypred, average='weighted')