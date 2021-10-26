
from nltk.probability import FreqDist
from sklearn.preprocessing import normalize
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import torch
from transformers import BertTokenizerFast, BertConfig, BertModel

import streamlit as st
import numpy as np


def getVectorizer(wordEmbedding, vec_dict, numLabels):
	if wordEmbedding == "tf-idf":
		vectorizer = TfidfVectorizer(strip_accents=vec_dict['strip_accents'], max_df=vec_dict['max_df'], min_df=vec_dict['min_df'], norm=vec_dict['norm'], stop_words='english')
		title = "tf-idf"

	if wordEmbedding == "bag-of-words":
		vectorizer = CountVectorizer(max_df=vec_dict['max_df'], min_df=vec_dict['min_df'])
		title = "bag-of-words"

	elif wordEmbedding == "word2vec":
		vectorizer = Word2Vec(vector_size=vec_dict['vector_size'], window=vec_dict['window'], min_count=vec_dict['min_count'], sg=vec_dict['sg'], hs=vec_dict['hs'], negative=vec_dict['negative'], cbow_mean=vec_dict['cbow_mean'], alpha=vec_dict['alpha'], min_alpha=vec_dict['min_alpha'], sample=vec_dict['sample'], epochs=vec_dict['epochs'])
		title = "word2vec"

	elif wordEmbedding == "doc2vec":
		vectorizer = Doc2Vec(vector_size=vec_dict['vector_size'], window=vec_dict['window'], min_count=vec_dict['min_count'], dm=vec_dict['dm'], hs=vec_dict['hs'], negative=vec_dict['negative'], ns_exponent=vec_dict['ns_exponent'], alpha=vec_dict['alpha'], min_alpha=vec_dict['min_alpha'], sample=vec_dict['sample'], epochs=vec_dict['epochs'], dbow_words=vec_dict['dbow_words'])
		title = "doc2vec"
	
	elif wordEmbedding == "BERT":
		config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
		vectorizer = BertModel.from_pretrained("bert-base-uncased", config=config)
		title = "BERT"
	
	elif wordEmbedding == "GloVe":
		vectorizer = []
		title = "GloVe"
	
	return vectorizer, title



def computeVec(vectorizer, tagvec, data, method4Doc):
	with st.spinner('Computing vectorization...'):
		if tagvec == "tf-idf":
			vec = vectorizer.fit_transform(data)
		elif tagvec == "bag-of-words":
			vec = vectorizer.fit_transform(data)
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
		elif tagvec == "doc2vec":
			taggedDocs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(data)]
			vectorizer.build_vocab(documents=taggedDocs)
			vectorizer.train(taggedDocs, total_examples=vectorizer.corpus_count, epochs=vectorizer.epochs)
			vec = np.array([vectorizer.docvecs[i] for i in range(len(taggedDocs))])
		elif tagvec == "BERT":
			input_ids, attention_masks = input4Bert(data)
			with torch.no_grad():
				outputs = vectorizer(input_ids, attention_mask=attention_masks)
				cls = get_CLSembeddingsBert(outputs)
				vec = cls[-1]
		elif tagvec == "GloVe":
			embedding_glove = loadGlove(data)
			vec = getMyEmbeddingOfGlove(embedding_glove, data)
		if issparse(vec):
			vec = vec.toarray()

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

def input4Bert(data):
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	input_ids = []
	attention_masks = []
	for sentence in data:
		dictionary = tokenizer.encode_plus(
							sentence,                      
							add_special_tokens = True,
							max_length = 30,
							pad_to_max_length = True,
							return_attention_mask = True,
							return_tensors = 'pt',
					)
		# encode_plus returns a dictionary 
		input_ids.append(dictionary['input_ids'])
		attention_masks.append(dictionary['attention_mask'])

	input_ids = torch.cat(input_ids, dim=0)
	attention_masks = torch.cat(attention_masks, dim=0)

	return input_ids, attention_masks

def get_CLS_embedding(layer):
    return layer[:, 0, :].numpy()

def get_CLSembeddingsBert(outputs):
	embeddings = outputs[2][1:]
	cls_embeddings = []
	for i in range(12):
		cls_embeddings.append(get_CLS_embedding(embeddings[i]))
	return cls_embeddings

def loadGlove(data):
	docsTokens = [val for doc in data for val in doc.split()]
	embeddings_dict = {}
	with open("models/glove.6B.100d.txt", 'r', encoding="UTF-8") as f:
		for line in f:
			values = line.split()
			word = values[0]
			if word in docsTokens:
				vector = np.asarray(values[1:], "float32")
				embeddings_dict[word] = vector
	return embeddings_dict

def getMyEmbeddingOfGlove(embeddings_glove, data):
	doc_features  = []
	for doc in data:
		aux = []
		docT = doc.split()
		for word in docT:
			if word in embeddings_glove:
				aux.append(embeddings_glove[word])
		doc_features.append(np.mean(aux, axis=0))
	return np.array(doc_features)

