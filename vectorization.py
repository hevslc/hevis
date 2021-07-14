
from nltk.probability import FreqDist
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import streamlit as st
import numpy as np


def getVectorizer(wordEmbedding, vec_dict):
	if wordEmbedding == "tf-idf":
		vectorizer = TfidfVectorizer(strip_accents=vec_dict['strip_accents'], max_df=vec_dict['max_df'], min_df=vec_dict['min_df'], norm=vec_dict['norm'], stop_words='english')
		title = "tf-idf"

	elif wordEmbedding == "word2vec":
		vectorizer = Word2Vec(vector_size=vec_dict['vector_size'], window=vec_dict['window'], min_count=vec_dict['min_count'], sg=vec_dict['sg'], hs=vec_dict['hs'], negative=vec_dict['negative'], cbow_mean=vec_dict['cbow_mean'], alpha=vec_dict['alpha'], min_alpha=vec_dict['min_alpha'], sample=vec_dict['sample'], epochs=vec_dict['epochs'])
		title = "word2vec"
	
	return vectorizer, title



def computeVec(vectorizer, tagvec, data, method4Doc, noesparse):
	with st.spinner('Computing vectorization...'):
		if tagvec == "tf-idf":
			vec = vectorizer.fit_transform(data)
			if noesparse:
				vec = vec.toarray()
		elif tagvec == "word2vec":
			vectorizer.build_vocab(data)
			vectorizer.train(data, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
			
			if method4Doc == 'sum_vectors':
				vec = sumVectors(vectorizer, data)

			if method4Doc == 'average_weighted_frequency':
				vec = averageWeightedFrequency(vectorizer, data)

			if method4Doc == 'add_and_norm':
				vec = addAndNorm(vectorizer, data)

			if method4Doc == 'add_and_tag':
				vec = addAndTag(vectorizer, data)	
	return vec


def sumVectors(vectorizer, data):
	vec = [np.sum([vectorizer.wv[word] for word in doc if word in vectorizer], axis=0) for doc in data]
	vec = np.vstack(vec) 
	return vec 

def averageWeightedFrequency(vectorizer, data):
	vecl = []
	for doc in data:
		fdist = FreqDist(word for word in doc)
		vecl.append(np.mean([fdist[word]*vectorizer.wv[word] for word in doc], axis=0))
		vec = np.vstack(vecl)
	return vec

def addAndNorm(vectorizer, data):
	vec = normalize(np.vstack(sumVectors(vectorizer, data)), axis=0)
	return vec

def addAndTag(vectorizer, data):
	tags = np.array([range(0,len(data))])
	vec = np.concatenate((sumVectors(vectorizer, data), tags.T), axis=1)
	return vec