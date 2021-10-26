
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA
import umap
import streamlit as st





def getProjection(projection, proj_dict, title):
	noesparse = False
	if projection == "PCA":
	    proj = PCA(whiten=proj_dict['whiten'], svd_solver=proj_dict['svd_solver'], n_components=2)
	    title = title + " and PCA"
	    noesparse = True
	elif projection == "t-SNE":
	    proj = TSNE(perplexity=proj_dict['perplexity'], learning_rate=proj_dict['learning_rate'], n_iter=proj_dict['n_iter'], n_iter_without_progress=proj_dict['n_iter_without_progress'], init=proj_dict['init'], method=proj_dict['method'])
	    title = title + " and t-SNE"
	    noesparse = True
	elif projection == "UMAP":
	    proj = umap.UMAP(metric=proj_dict['metric'], n_neighbors=proj_dict['n_neighbors'], n_epochs =proj_dict['n_epochs'], learning_rate =proj_dict['learning_rate'], init =proj_dict['init'], min_dist =proj_dict['min_dist'], spread =proj_dict['spread'], set_op_mix_ratio =proj_dict['set_op_mix_ratio'])
	    title = title + " and UMAP"
	    noesparse = True
	elif projection == "Isomap":
	    proj = Isomap(n_neighbors=proj_dict['n_neighbors'], eigen_solver=proj_dict['eigen_solver'], path_method=proj_dict['path_method'], neighbors_algorithm=proj_dict['neighbors_algorithm'], p=proj_dict['p'])
	    title = title + " and Isomap"
	elif projection == "LocallyLinearEmbedding":
	    proj = LocallyLinearEmbedding(n_neighbors=proj_dict['n_neighbors'], eigen_solver=proj_dict['eigen_solver'], method=proj_dict['method'], neighbors_algorithm=proj_dict['neighbors_algorithm'])
	    title = title + " and Locally Linear Embedding"

	return proj, title, noesparse


def computeProj(proj, vec_wembedding):
	with st.spinner('Computing the projection...'):		
		vec_proj = proj.fit_transform(vec_wembedding)
	return vec_proj	

