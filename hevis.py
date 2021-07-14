#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import glob
import streamlit as st


import auxfuncs as aux
import vectorization as veclib
import projection as projlib

from streamlit_plotly_events import plotly_events

nsamples = 100000
title = "Projection"

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 400px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 400px;
        margin-left: -500px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

preprocessedFile = glob.glob("data/preprocessed.csv", recursive = True)
originalText, ppText, target = aux.default(bool(len(preprocessedFile)))
noesparse = False

labels = []
labelsFile = glob.glob("data/labels.txt", recursive = True)
if bool(len(labelsFile)):
    with open('data/labels.txt') as f:
        labels = [line.rstrip() for line in f]


bestneigh = aux.avgAmountLabels(target)

# Side Bar #######################################################

#uploaded_file = st.sidebar.file_uploader("Choose a file")
#if uploaded_file is not None:
    #bytes_data = uploaded_file.getvalue() # To read file as bytes
    #st.write(bytes_data)
    
vec_dict = {}
proj_dict = {}
vec_dict['method4Doc'] = None


st.sidebar.markdown("## Word Embeddings")
wordEmbedding = st.sidebar.selectbox((''), ('tf-idf', 'word2vec'))
if wordEmbedding == "tf-idf":
    form_vec = st.sidebar.form("config-word-embedding")
    vec_dict['strip_accents'] = form_vec.selectbox(('strip_accents'), (None, 'ascii', 'unicode'))
    vec_dict['max_df'] = form_vec.slider('max_df', 0.0, 1.0, 1.0)
    vec_dict['min_df'] = form_vec.slider('min_df', 0, 100, 1)
    vec_dict['norm'] = form_vec.selectbox(('norm'), ('l1', 'l2'), index=1)
    submitted_vec = form_vec.form_submit_button("OK")
elif wordEmbedding == "word2vec":
    form_vec = st.sidebar.form("config-word-embedding")
    vec_dict['vector_size'] = form_vec.slider('vector_size', 0, 10000, 100)
    vec_dict['window'] = form_vec.slider('window', 0, 50, 5)
    vec_dict['min_count'] = form_vec.slider('min_count', 1, 100, 1)
    vec_dict['sg'] = form_vec.slider('sg', 0, 1, 0)
    vec_dict['hs'] = form_vec.slider('hs', 0, 1, 0)
    vec_dict['negative'] = form_vec.slider('negative', 0, 20, 5)
    vec_dict['cbow_mean'] = form_vec.slider('cbow_mean', 0, 1, 1)
    vec_dict['alpha'] = form_vec.slider('alpha', 0.0, 1.0, 0.025)
    vec_dict['min_alpha'] = form_vec.slider('min_alpha', 0.0, 1.0, 0.0001)
    vec_dict['sample'] = form_vec.slider('sample', 0.0, 0.1, 0.001)
    vec_dict['epochs'] = form_vec.slider('epochs', 1, 100, 5)
    vec_dict['method4Doc'] = form_vec.selectbox(('method4Doc'), ('sum_vectors', 'average_weighted_frequency', 'add_and_norm'), index=1)
    submitted_vec = form_vec.form_submit_button("OK")


st.sidebar.markdown("## Projections")
projection = st.sidebar.selectbox((''), ('PCA', 't-SNE', 'MDS', 'Isomap', 'SpectralEmbedding'))
if projection == "PCA":
    form_proj = st.sidebar.form("config-proj")
    proj_dict['whiten'] = form_proj.selectbox(('whiten'), ('False', 'True'))
    proj_dict['svd_solver'] = form_proj.selectbox(('svd_solver'), ('auto', 'full', 'arpack', 'randomized'))
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "t-SNE":
    form_proj = st.sidebar.form("config-proj")
    proj_dict['perplexity'] = form_proj.slider('perplexity', 5.0, 50.0, 30.0)
    proj_dict['learning_rate']  = form_proj.slider('learning_rate ', 10.0, 1000.0, 200.0)
    proj_dict['n_iter']  = form_proj.slider('n_iter ', 250, 2000, 1000)
    proj_dict['n_iter_without_progress']  = form_proj.slider('n_iter_without_progress ', 250, 2000, 1000)
    proj_dict['init'] = form_proj.selectbox(('init'), ('random', 'pca'), index=1)
    proj_dict['method'] = form_proj.selectbox(('method'), ('barnes_hut', 'exact'))
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "MDS":
    form_proj = st.sidebar.form("config-proj")
    proj_dict['metric'] = form_proj.selectbox(('metric'), (True, False))
    proj_dict['n_init'] = form_proj.slider('n_init', 1, 10, 4)  
    proj_dict['max_iter']  = form_proj.slider('max_iter ', 50, 1000, 300)
    #proj_dict['eps']  = form_proj.slider('eps ', 0.0000000001, 0.1, 0.001, step=0.00000000001)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "Isomap":
    proj_dict['form_proj'] = st.sidebar.form("config-proj")
    proj_dict['n_neighbors'] = form_proj.slider(f'n_neighbors - Try    {bestneigh}', 1, 1000, 5)    
    proj_dict['eigen_solver'] = form_proj.selectbox(('eigen_solver'), ('auto', 'arpack', 'dense'))
    proj_dict['path_method'] = form_proj.selectbox(('path_method'), ('auto', 'FW', 'D'))
    proj_dict['neighbors_algorithm'] = form_proj.selectbox(('neighbors_algorithm'), ('auto', 'brute', 'kd_tree', 'ball_tree'))
    proj_dict['p'] = form_proj.slider('p', 1, 10, 2)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "SpectralEmbedding":
    proj_dict['form_proj'] = st.sidebar.form("config-proj")
    proj_dict['affinity'] = form_proj.selectbox(('affinity'), ('nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'))
    proj_dict['eigen_solver'] = form_proj.selectbox(('eigen_solver'), ('arpack', 'lobpcg', 'amg'))
    proj_dict['n_neighbors'] = form_proj.slider('n_neighbors', 1, nsamples, int(nsamples/10))   
    submitted_proj = form_proj.form_submit_button("OK")  

vectorizer, title = veclib.getVectorizer(wordEmbedding, vec_dict)
proj, title, noesparse = projlib.getProjection(projection, proj_dict, title)



# App ##################################################

st.title('Visualization')


if submitted_vec or submitted_proj:
    aux.compute(vectorizer, wordEmbedding, vec_dict['method4Doc'], proj, ppText, target, labels, noesparse)
    

fig_js, mapp, dataproj, datavec = aux.readData()

neighpreserv, coefsil = aux.layoutQuality(datavec, dataproj, target)
st.header(f'Neighborhood Preservation: {neighpreserv:.2f}    Coeficiente Silhouette: {coefsil:.2f}' )

plot_name_holder = st.empty()
clickedPoint = plotly_events(fig_js, key='scatter')
#plot_name_holder.write(f"Clicked Point: {clickedPoint}")

with st.beta_expander('Text'):
    if len(clickedPoint):
        idx = aux.getTextIdx(mapp, clickedPoint[0]['curveNumber'], clickedPoint[0]['pointIndex'])
        st.subheader("Texto pré-processado")
        st.write(ppText[idx])
        st.subheader("Texto original")
        st.text(originalText[idx])
