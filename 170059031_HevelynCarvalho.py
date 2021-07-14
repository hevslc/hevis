#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import glob
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA

import auxfuncs as aux

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
data, target = aux.default(bool(len(preprocessedFile)))
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


form_tfidf = st.sidebar.form("config-tfidf")
form_tfidf.markdown("## tf-idf")
strip_accents = form_tfidf.selectbox(('strip_accents'), (None, 'ascii', 'unicode'))
max_df = form_tfidf.slider('max_df', 0.0, 1.0, 1.0)
min_df = form_tfidf.slider('min_df', 0, 100, 1)
norm = form_tfidf.selectbox(('norm'), ('l1', 'l2'), index=1)
submitted_tfidf = form_tfidf.form_submit_button("OK")


st.sidebar.markdown("## Projections")
projection = st.sidebar.selectbox(('projection'), ('PCA', 't-SNE', 'MDS', 'Isomap', 'SpectralEmbedding'))
if projection == "PCA":
    form_proj = st.sidebar.form("config-proj")
    whiten = form_proj.selectbox(('whiten'), ('False', 'True'))
    svd_solver = form_proj.selectbox(('svd_solver'), ('auto', 'full', 'arpack', 'randomized'))
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "t-SNE":
    form_proj = st.sidebar.form("config-proj")
    perplexity = form_proj.slider('perplexity', 5.0, 50.0, 30.0)
    learning_rate  = form_proj.slider('learning_rate ', 10.0, 1000.0, 200.0)
    n_iter  = form_proj.slider('n_iter ', 250, 2000, 1000)
    n_iter_without_progress  = form_proj.slider('n_iter_without_progress ', 250, 2000, 1000)
    init = form_proj.selectbox(('init'), ('random', 'pca'), index=1)
    method = form_proj.selectbox(('method'), ('barnes_hut', 'exact'))
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "MDS":
    form_proj = st.sidebar.form("config-proj")
    metric = form_proj.selectbox(('metric'), (True, False))
    n_init = form_proj.slider('n_init', 1, 10, 4)  
    max_iter  = form_proj.slider('max_iter ', 50, 1000, 300)
    #eps  = form_proj.slider('eps ', 0.0000000001, 0.1, 0.001, step=0.00000000001)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "Isomap":
    form_proj = st.sidebar.form("config-proj")
    n_neighbors = form_proj.slider(f'n_neighbors - Try    {bestneigh}', 1, 1000, 5)    
    eigen_solver = form_proj.selectbox(('eigen_solver'), ('auto', 'arpack', 'dense'))
    path_method = form_proj.selectbox(('path_method'), ('auto', 'FW', 'D'))
    neighbors_algorithm = form_proj.selectbox(('neighbors_algorithm'), ('auto', 'brute', 'kd_tree', 'ball_tree'))
    p = form_proj.slider('p', 1, 10, 2)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "SpectralEmbedding":
    form_proj = st.sidebar.form("config-proj")
    affinity = form_proj.selectbox(('affinity'), ('nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'))
    eigen_solver = form_proj.selectbox(('eigen_solver'), ('arpack', 'lobpcg', 'amg'))
    n_neighbors = form_proj.slider('n_neighbors', 1, nsamples, int(nsamples/10))   
    submitted_proj = form_proj.form_submit_button("OK")  


vectorizer = TfidfVectorizer(strip_accents=strip_accents, max_df=max_df, min_df=min_df, norm=norm, stop_words='english')

if projection == "PCA":
    proj = PCA(whiten=whiten, svd_solver=svd_solver, n_components=2)
    title = "(PCA)"
    noesparse = True
elif projection == "t-SNE":
    proj = TSNE(perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, n_iter_without_progress=n_iter_without_progress, init=init, method=method)
    title = "(t-SNE)"
    noesparse = True
elif projection == "MDS":
    proj = MDS(metric=metric, n_init=n_init, max_iter =max_iter)
    title = "(MDS)"
    noesparse = True
elif projection == "Isomap":
    proj = Isomap(n_neighbors=n_neighbors, eigen_solver=eigen_solver, path_method=path_method, neighbors_algorithm=neighbors_algorithm, p=p)
    title = "(Isomap)"
elif projection == "SpectralEmbedding":
    proj = SpectralEmbedding(affinity=affinity, eigen_solver=eigen_solver, n_neighbors=n_neighbors)
    title = "(Spectral Embedding)"



# App ##################################################

st.title('Visualization')


if submitted_tfidf or submitted_proj:
    aux.compute(vectorizer, proj, data, target, labels, noesparse)


fig_js, mapp, dataproj, datavec = aux.readData()

neighpreserv, coefsil = aux.layoutQuality(datavec, dataproj, target)
st.header(f'Neighborhood Preservation: {neighpreserv:.2f}    Coeficiente Silhouette: {coefsil:.2f}' )
#st.plotly_chart(fig)

plot_name_holder = st.empty()
clickedPoint = plotly_events(fig_js, key='scatter')
plot_name_holder.write(f"Clicked Point: {clickedPoint}")
   




