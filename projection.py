
from sklearn.manifold import MDS, TSNE, Isomap, SpectralEmbedding
from sklearn.decomposition import TruncatedSVD, PCA
import streamlit as st





def getProjection(projection, proj_dict, title):
	if projection == "PCA":
	    proj = PCA(whiten=proj_dict['whiten'], svd_solver=proj_dict['svd_solver'], n_components=2)
	    title = title + "(PCA)"
	    noesparse = True
	elif projection == "t-SNE":
	    proj = TSNE(perplexity=proj_dict['perplexity'], learning_rate=proj_dict['learning_rate'], n_iter=proj_dict['n_iter'], n_iter_without_progress=proj_dict['n_iter_without_progress'], init=proj_dict['init'], method=proj_dict['method'])
	    title = title + "(t-SNE)"
	    noesparse = True
	elif projection == "MDS":
	    proj = MDS(metric=proj_dict['metric'], n_init=proj_dict['n_init'], max_iter =proj_dict['max_iter'])
	    title = title + "(MDS)"
	    noesparse = True
	elif projection == "Isomap":
	    proj = Isomap(n_neighbors=proj_dict['n_neighbors'], eigen_solver=proj_dict['eigen_solver'], path_method=proj_dict['path_method'], neighbors_algorithm=proj_dict['neighbors_algorithm'], p=proj_dict['p'])
	    title = title + "(Isomap)"
	elif projection == "SpectralEmbedding":
	    proj = SpectralEmbedding(affinity=proj_dict['affinity'], eigen_solver=proj_dict['eigen_solver'], n_neighbors=proj_dict['n_neighbors'])
	    title = title + "(Spectral Embedding)"

	return proj, title, noesparse


def computeProj(proj, vec_wembedding):
	with st.spinner('Computing the projection...'):		
		vec_proj = proj.fit_transform(vec_wembedding)
	return vec_proj	