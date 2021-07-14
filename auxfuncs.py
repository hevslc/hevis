import pandas as pd
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import vectorization as veclib
import projection as projlib
import plotfig as plotfiglib


import pickle
import json


# Avaliação das projeções
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score

nltk.download('wordnet')
nltk.download('punkt')


def compute(vectorizer, tagvec, method4Doc, proj, data, target, labels, noesparse):
	vec_wembedding = veclib.computeVec(vectorizer, tagvec, data, method4Doc, noesparse)
	vec_proj = projlib.computeProj(proj, vec_wembedding)
	fig, mapp = plotfiglib.computeFig(vec_proj, target, labels)


	# Save in tmp/
	with open('tmp/vec_wembedding', 'wb') as f:
	    pickle.dump(vec_wembedding, f)
	with open('tmp/vec_proj', 'wb') as f:
		pickle.dump(vec_proj, f)
	with open('tmp/mapp', 'wb') as f:
	    pickle.dump(mapp, f)		
	with open('tmp/fig.json', 'w') as f:
		json.dump(fig.to_json(), f)



def readData():
	with open ('tmp/vec_proj', 'rb') as f:
	    vec_proj = pickle.load(f)
	with open ('tmp/vec_wembedding', 'rb') as f:
	    vec_wembedding = pickle.load(f)
	with open ('tmp/mapp', 'rb') as f:
	    mapp = pickle.load(f)	    
	with open('tmp/fig.json', 'r') as j:
	    fig = json.load(j)
	return fig, mapp, vec_proj, vec_wembedding	


def preprocessing(text):
	text = text.lower()
	tokens = word_tokenize(text)
	words = [WordNetLemmatizer().lemmatize(w) for w in tokens if w.isalpha()]
	return TreebankWordDetokenizer().detokenize(words)

def getTextIdx(mapp, curveNumber, pointIndex):
	return mapp[curveNumber][pointIndex]


def default(pp):
	data = pd.read_csv('data/data.csv',encoding="utf8")
	originalText = data["text"]
	if(pp):
		df_datapp = pd.read_csv('data/preprocessed.csv',encoding="utf8")
	else:
		datapp = list(map(preprocessing, data["text"]))
		df_datapp = data
		df_datapp['text'] = datapp
		df_datapp.to_csv('data/preprocessed.csv', index=False)  

	return originalText, df_datapp['text'], data['label']

def avgAmountLabels(labels):
	c = Counter(labels)
	return int(sum(c.values())/len(c))

def layoutQuality(original, actual, labels):
	m = avgAmountLabels(labels)
	neighpreservation = trustworthiness(original, actual, n_neighbors=int(m+10), metric='euclidean')
	sil = silhouette_score(actual, labels, sample_size=int(m+10), metric='euclidean')
	return neighpreservation, sil