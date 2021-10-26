import glob
import os
import numpy as np
from collections import Counter
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

import src.vectorization as veclib
import src.projection as projlib
import src.plotfig as plotfiglib


import pickle
import json
from datasets import load_dataset
from sklearn.datasets import fetch_20newsgroups

import streamlit as st

# Avaliação das projeções
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
import time

nltk.download('wordnet')
nltk.download('punkt')

def createtmp():
	tmp = glob.glob("tmp/results")
	if(not bool(len(tmp))):
		os.mkdir("tmp/results")
		os.mkdir("tmp/results/vecs_embeddings")
		os.mkdir("tmp/results/vecs_projections")
		os.mkdir("tmp/results/neighMetrics")
		os.mkdir("tmp/results/silMetrics")
		os.mkdir("tmp/results/f1Scores")

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
	file = glob.glob("tmp/vec_wembedding.npy")
	if not bool(len(file)):	
		_ = compute(True,vectorizer, tagvec, method4Doc, proj, data, target, labels)


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


def default():
	'''dataset = load_dataset('amazon_polarity', split='test')
	data = dataset.to_pandas()
	originalText = data["content"]
	target = data["label"]
	labels = dataset.features['label'].names

	originalText = originalText[:7000]
	target = target[:7000]'''

	'''dataset = load_dataset('ag_news', split='test')  
	data = dataset.to_pandas()
	originalText = data["text"]
	target = data["label"]
	labels = dataset.features['label'].names '''  

	remove = ('headers', 'footers', 'quotes')
	categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc',
 				'comp.windows.x', 'misc.forsale', 'rec.autos']
	dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=0, remove=remove, categories=categories)
	originalText = dataset.data
	target = dataset.target
	labels = dataset.target_names


	pp = bool(len(glob.glob("tmp/preprocessed")))
	if(pp):
		with open ('tmp/preprocessed', 'rb') as f:
			datapp = pickle.load(f)
	else:
		with st.spinner('Pre-processing...'):
			datapp = list(map(preprocessing, originalText))
			with open('tmp/preprocessed', 'wb') as f:
				pickle.dump(datapp, f)
	
	return labels, originalText, datapp, target

def avgAmountLabels(labels):
	c = Counter(labels)
	return int(sum(c.values())/len(c))

@st.cache
def getTrustworthiness(original, actual, n):
	return trustworthiness(original, actual, n_neighbors=n)


@st.cache
def listTrustworthiness(original, actual, kneigh):
	neighpreservation_list = []
	for n in range(1,kneigh + 1):
		neighpreservation_list.append(trustworthiness(original, actual, n_neighbors=n))
	return neighpreservation_list

@st.cache
def getSilhouetteCoefficient(actual, labels):
	return (silhouette_score(actual, labels))

def getFileName(title):
	filename = title.replace(" ", "")
	filename = filename.replace("-", "")
	filename = filename.replace("+", "_")
	filename = filename.replace("and", "_")
	return filename	

def saveResults(title, vec_embedding, vec_proj, neighMetric, silMetric, f1score):
	filename = getFileName(title)
	np.save('tmp/results/vecs_embeddings/' + filename, vec_embedding)
	np.save('tmp/results/vecs_projections/' + filename, vec_proj)
	np.save('tmp/results/neighMetrics/' + filename, neighMetric)
	np.save('tmp/results/silMetrics/' + filename, silMetric)
	np.save('tmp/results/f1Scores/' + filename, f1score)

def history():
	files_vec_proj = glob.glob('tmp/results/vecs_projections/*')
	files_neighMetric = glob.glob('tmp/results/neighMetrics/*')
	files_silMetric = glob.glob('tmp/results/silMetrics/*')
	files_f1score = glob.glob('tmp/results/f1Scores/*')

	neighCycles = readFromFiles(files_neighMetric)
	silCycles = readFromFiles(files_silMetric)
	f1scores = readFromFiles(files_f1score)


	titles = list(map(getTitle, files_neighMetric))

	vec_neigh = getVecNeigh(neighCycles)
	vec_sil = getVecSil(silCycles)
	vec_f1score = getVecF1score(f1scores)
	historyValues = plotfiglib.plotHistory(titles, vec_sil, vec_neigh, vec_f1score)
	return historyValues

def readFromFiles(file_names):
	l = []
	for file in file_names:
		l.append(np.load(file))
	return l

def getTitle(fileName):
	fileName = fileName.split(".")[0]
	fileName = fileName.split("/")[-1]
	fileName = fileName.split("\\")[-1]
	return fileName

def getVecNeigh(cycles):
	vec_neigh = np.array(cycles).flatten()
	return vec_neigh
	
def getVecSil(cycles):
	vec_sil = np.array(cycles).flatten()
	return vec_sil

def getVecF1score(cycles):
	vec_f1score = np.array(cycles).flatten()
	return vec_f1score