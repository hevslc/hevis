#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import glob
import streamlit as st


import auxfuncs as aux
import vectorization as veclib
import projection as projlib
import plotfig as plot_lib
import classification as clf

from streamlit_plotly_events import plotly_events


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

preprocessedFile = glob.glob("data/preprocessed", recursive = True)
labels, originalText, ppText, target = aux.default(bool(len(preprocessedFile)))
noesparse = False
nsamples = len(target)


bestneigh = aux.avgAmountLabels(target)

# Side Bar #######################################################

#uploaded_file = st.sidebar.file_uploader("Choose a file")
#if uploaded_file is not None:
    #bytes_data = uploaded_file.getvalue() # To read file as bytes
    #st.write(bytes_data)
    
vec_dict = {}
proj_dict = {}
vec_dict['method4Doc'] = "average_weighted_frequency"


st.sidebar.markdown("## Word Embeddings")
wordEmbedding = st.sidebar.selectbox((''), ('tf-idf', 'word2vec', 'bag-of-words', "doc2vec", "BERT", "GloVe"))
if wordEmbedding == "tf-idf":
    form_vec = st.sidebar.form("config-word-embedding")
    vec_dict['sublinear_tf'] = form_vec.selectbox(('sublinear_tf'), ('False', 'True'))
    vec_dict['strip_accents'] = form_vec.selectbox(('strip_accents'), (None, 'ascii', 'unicode'))
    vec_dict['max_df'] = form_vec.slider('max_df', 0.0, 1.0, 1.0)
    vec_dict['min_df'] = form_vec.slider('min_df', 0.0, 1.0, 0.0)
    vec_dict['norm'] = form_vec.selectbox(('norm'), ('l1', 'l2'), index=1)
    submitted_vec = form_vec.form_submit_button("OK")
elif wordEmbedding == "bag-of-words":
    form_vec = st.sidebar.form("config-word-embedding")
    vec_dict['max_df'] = form_vec.slider('max_df', 0.0, 1.0, 1.0)
    vec_dict['min_df'] = form_vec.slider('min_df', 0.0, 1.0, 0.2)
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
elif wordEmbedding == "doc2vec":
    form_vec = st.sidebar.form("config-word-embedding")
    vec_dict['vector_size'] = form_vec.slider('vector_size', 0, 10000, 100)
    vec_dict['window'] = form_vec.slider('window', 0, 50, 5)
    vec_dict['min_count'] = form_vec.slider('min_count', 1, 100, 1)
    vec_dict['hs'] = form_vec.slider('hs', 0, 1, 0)
    vec_dict['dm'] = form_vec.slider('dm', 0, 1, 1)
    vec_dict['negative'] = form_vec.slider('negative', 0, 20, 5)
    vec_dict['ns_exponent'] = form_vec.slider('ns_exponent', 0.0, 1.0, 0.75)
    vec_dict['alpha'] = form_vec.slider('alpha', 0.0, 1.0, 0.025)
    vec_dict['min_alpha'] = form_vec.slider('min_alpha', 0.0, 1.0, 0.0001)
    vec_dict['sample'] = form_vec.slider('sample', 0.0, 0.1, 0.001)
    vec_dict['epochs'] = form_vec.slider('epochs', 1, 100, 10)
    vec_dict['dbow_words'] = form_vec.slider('dbow_words', 0, 1, 0)
    submitted_vec = form_vec.form_submit_button("OK")
elif wordEmbedding == "BERT":
    form_vec = st.sidebar.form("config-word-embedding")
    submitted_vec = form_vec.form_submit_button("OK")
elif wordEmbedding == "GloVe":
    form_vec = st.sidebar.form("config-word-embedding")
    submitted_vec = form_vec.form_submit_button("OK")


st.sidebar.markdown("## Projections")
projection = st.sidebar.selectbox((''), ('PCA', 't-SNE', 'UMAP', 'Isomap', 'LocallyLinearEmbedding'))
if projection == "PCA":
    noesparse = True
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
elif projection == "UMAP":
    form_proj = st.sidebar.form("config-proj")
    proj_dict['n_neighbors'] = form_proj.slider(f'n_neighbors - Try    {bestneigh}', 2, 100, 15)    
    proj_dict['metric'] = form_proj.selectbox(('metric'), ("euclidean", "manhattan", "chebyshev", "minkowski", "canberra", "braycurtis", "mahalanobis", "wminkowski", "seuclidean", "cosine", "correlation", "haversine", "hamming", "jaccard", "dice", "russelrao", "kulsinski", "ll_dirichlet", "hellinger", "rogerstanimoto", "sokalmichener", "sokalsneath", "yule"))
    proj_dict['n_epochs'] = form_proj.slider('n_epochs', 1, 2000, 500)  
    proj_dict['learning_rate']  = form_proj.slider('learning_rate ', 0.0, 1.0, 1.0)
    proj_dict['init'] = form_proj.selectbox(("init"), ("spectral", "random"))
    proj_dict['min_dist']  = form_proj.slider('min_dist', 0.0, 1.0, 1.0)
    proj_dict['spread']  = form_proj.slider('spread', 0.0, 1.0, 1.0)
    proj_dict['set_op_mix_ratio']  = form_proj.slider('set_op_mix_ratio', 0.0, 1.0, 1.0)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "Isomap":
    form_proj = st.sidebar.form("config-proj")
    proj_dict['n_neighbors'] = form_proj.slider(f'n_neighbors - Try    {bestneigh}', 1, 50, 5)    
    proj_dict['eigen_solver'] = form_proj.selectbox(('eigen_solver'), ('auto', 'arpack', 'dense'))
    proj_dict['path_method'] = form_proj.selectbox(('path_method'), ('auto', 'FW', 'D'))
    proj_dict['neighbors_algorithm'] = form_proj.selectbox(('neighbors_algorithm'), ('auto', 'brute', 'kd_tree', 'ball_tree'))
    proj_dict['p'] = form_proj.slider('p', 1, 10, 2)
    submitted_proj = form_proj.form_submit_button("OK")
elif projection == "LocallyLinearEmbedding":
    noesparse = True
    form_proj = st.sidebar.form("config-proj")
    proj_dict['n_neighbors'] = form_proj.slider('n_neighbors', 1, 100, 5, 1)  
    proj_dict['eigen_solver'] = form_proj.selectbox(('eigen_solver'), ('auto', 'arpack', 'dense'))
    proj_dict['method'] = form_proj.selectbox(('method'), ('standard', 'hessian', 'modified', 'ltsa'))
    proj_dict['neighbors_algorithm'] = form_proj.selectbox(('neighbors_algorithm'), ('auto', 'brute', 'kd_tree', 'ball_tree'))
    submitted_proj = form_proj.form_submit_button("OK")  

vectorizer, title = veclib.getVectorizer(wordEmbedding, vec_dict, len(np.unique(target)))
proj, title, noesparse = projlib.getProjection(projection, proj_dict, title)



# App ##################################################
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#ectorizer = Doc2Vec(vector_size=vec_dict['vector_size'], window=vec_dict['window'], min_count=vec_dict['min_count'], dm=vec_dict['dm'], hs=vec_dict['hs'], negative=vec_dict['negative'], ns_exponent=vec_dict['ns_exponent'], alpha=vec_dict['alpha'], min_alpha=vec_dict['min_alpha'], sample=vec_dict['sample'], epochs=vec_dict['epochs'], dbow_words=vec_dict['dbow_words'])
#aggedDocs = [TaggedDocument(doc.split(), np.array([i])) for i, doc in enumerate(ppText)]
#vectorizer.build_vocab(documents=taggedDocs)


vec_time, proj_time = 0, 0
if submitted_vec or submitted_proj:
    vec_time, proj_time = aux.compute(submitted_vec, vectorizer, wordEmbedding, vec_dict['method4Doc'], proj, ppText, target, labels)
    

fig_js, mapp, dataproj, datavec = aux.readData(vectorizer, wordEmbedding, vec_dict['method4Doc'], proj, ppText, target, labels)


np.save('tmp/label', target)

st.title('Visualization')
st.subheader(title)

clickedPoint = plotly_events(fig_js, key='scatter')

# neighpreservation_list, coefsil = aux.layoutQuality(datavec, dataproj, target)
# st.write(f'Coeficiente Silhouette: {coefsil:.2f}' )
# st.plotly_chart(plot_lib.lineChart(list(range(1,31)), neighpreservation_list, "Neighborhood Preservation", "Number Neighbors", "Precision"))


# f1score = clf.runSVC(dataproj, target)
# st.subheader(f'Classification f1-score: {f1score:.2f}')



st.subheader('Click on a point to view the text:')
if len(clickedPoint):
    idx = aux.getTextIdx(mapp, clickedPoint[0]['curveNumber'], clickedPoint[0]['pointIndex'])
    with st.beta_expander("Texto pré-processado"):
        st.write(ppText[idx])
    with st.beta_expander("Texto original"):
        st.text(originalText[idx])
