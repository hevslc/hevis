import glob
import numpy as np
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
from datasets import load_dataset

import streamlit as st

# Avaliação das projeções
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
import time

nltk.download('wordnet')
nltk.download('punkt')


def compute(submitted_vec, vectorizer, tagvec, method4Doc, proj, data, target, labels):
	t1 = time.time()
	if submitted_vec: 
		vec_wembedding = veclib.computeVec(vectorizer, tagvec, data, method4Doc)
	else:
		vec_wembedding = np.load('tmp/vec_wembedding.npy')
	t2 = time.time()
	vec_proj = projlib.computeProj(proj, vec_wembedding)
	t3 = time.time()
	fig, mapp = plotfiglib.computeFig(vec_proj, target, labels)


	# Save in tmp/
	if submitted_vec:
		np.save('tmp/vec_wembedding', vec_wembedding)
	
	np.save('tmp/vec_proj', vec_proj)
	np.save('tmp/mapp', np.array(mapp))

	with open('tmp/fig.json', 'w') as f:
		json.dump(fig.to_json(), f)
	
	return (t2-t1), (t3-t2)



def readData(vectorizer, tagvec, method4Doc, proj, data, target, labels):
	files = glob.glob("tmp/*", recursive = True)
	if not bool(len(files)):	
		_ = compute(False,vectorizer, tagvec, method4Doc, proj, data, target, labels)

	vec_proj = np.load('tmp/vec_proj.npy')
	vec_wembedding = np.load('tmp/vec_wembedding.npy')
	mapp = np.load('tmp/mapp.npy', allow_pickle=True)

	    
	with open('tmp/fig.json', 'r') as j:
	    fig = json.load(j)

	return fig, mapp, vec_proj, vec_wembedding	


def preprocessing(text):
	text = text.lower()
	tokens = word_tokenize(text)
	words = [WordNetLemmatizer().lemmatize(w) for w in tokens if w.isalpha()]
	return TreebankWordDetokenizer().detokenize(words)

def getTextIdx(mapp, curveNumber, pointIndex):
	return mapp[curveNumber][0][pointIndex]


def default(pp):
	'''dataset = load_dataset('amazon_polarity', split='test')
	data = dataset.to_pandas()
	originalText = data["content"]
	target = data["label"]
	labels = dataset.features['label'].names

	originalText = originalText[:7000]
	target = target[:7000]'''

	dataset = load_dataset('ag_news', split='test')  
	data = dataset.to_pandas()
	originalText = data["text"]
	target = data["label"]
	labels = dataset.features['label'].names   

	'''dataset = load_dataset('fake_news_english', split='test')
	data = dataset.to_pandas()
	originalText = data["url_of_article"]
	data["label"] = data["fake_or_satire"]
	labels = dataset.features['label'].names'''


	if(pp):
		with open ('data/preprocessed', 'rb') as f:
			datapp = pickle.load(f)
	else:
		with st.spinner('Pre-processing...'):
			datapp = list(map(preprocessing, originalText))
			with open('data/preprocessed', 'wb') as f:
				pickle.dump(datapp, f)
	
	return labels, originalText, datapp, target

def avgAmountLabels(labels):
	c = Counter(labels)
	return int(sum(c.values())/len(c))

@st.cache
def layoutQuality(original, actual, labels):
	neighpreservation_list = []
	for n in range(1,31):
		neighpreservation_list.append(trustworthiness(original, actual, n_neighbors=n))
	sil = (silhouette_score(actual, labels))
	return neighpreservation_list, sil