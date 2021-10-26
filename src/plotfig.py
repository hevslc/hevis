import plotly.graph_objs as go
from plotly import tools
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from json import loads

def saveImage(fig_js, filename, strformat):
	fig = go.FigureWidget(loads(fig_js))
	name = "downloads/trustworthiness_" + filename + "." + strformat
	return fig.write_image(name)

def formatColor(color):
	return 'rgb('+str(int(color[0]*255))+','+str(int(color[1]*255))+','+str(int(color[2]*255))+')'

def getTraces(X, Y, labels):
	numColors = len(np.unique(Y))
	cmap = plt.cm.get_cmap('hsv', (numColors+1)) #Maps each index in 0, 1, ..., n-1 to a distinct RGB color
	colorList = [cmap(i) for i in range(1, numColors+1)]


	traceArr = []
	mapPoints = []
	labelArray = np.unique(Y)
	if(len(labels)==0):
		for i in labelArray:
			labels.append("trace " + str(i))
	for lab, col in zip(labelArray,colorList):
		mapPoints.append(np.where(Y==lab))
		trace = go.Scatter(
			x=X[Y==lab, 0],
			y=X[Y==lab, 1],
			mode='markers',
			name = str(labels[lab-1]),
			marker=dict(
				size=5,
				color=formatColor(col),
				opacity=0.8
			),

		)
		traceArr.append(trace)
	return traceArr, mapPoints



def Fig(X, Y, labels):
	traces, mapp = getTraces(X, Y, labels)
	f = go.FigureWidget(data = traces)
	return f, mapp


def computeFig(vec_proj, target, labels):
	with st.spinner('Computing the figure...'):	
		fig, mapp = Fig(vec_proj, target, labels)	
	return fig, mapp

def lineChart(x, y, title, xtitle, ytitle):
	fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines+markers'))
	fig.update_layout(title=title, xaxis_title=xtitle, yaxis_title=ytitle)
	return fig

def plotSubplotsHistory(vecs, target, labels, titles, nrows, ncols):
	projs = []
	for v in vecs:
		projs.append(getTraces(v, target, labels))
	lenProjs = len(projs)
	fig = tools.make_subplots(rows=nrows, cols=ncols, subplot_titles=titles)
	idx = 0
	for i in range(1,(nrows+1)):
		j = 1
		while idx < lenProjs:
			if j <= 5:
				for k in range(len(labels)):
					fig.append_trace(projs[idx][0][k], i, j)
				idx = idx + 1
				j = j + 1
			else:
				break

	for data in fig.data:
  		data.update(showlegend=False)		
	for k in range(len(labels)):
  		fig.data[k].update(showlegend=True)
	fig.update_layout(title="Projections")
	
	return fig

def plotNeighValues(titles, neighVecs):
	nneigh = list(range(1,31))

	fig = go.Figure()
	for i,l in enumerate(neighVecs):
		fig.add_trace(go.Scatter(x=nneigh, y=l, mode='lines+markers', showlegend=False, name=titles[i]))

	annotations = []
	for y_trace, label in zip(neighVecs, titles):
		# labeling the right_side of the plot
		y = y_trace[-1]
		annotations.append(dict(xref='paper', x=0.86, y=y,
									xanchor='left', yanchor='middle',
									text=label,
									font=dict(family='Arial',
												size=9),
									showarrow=False))

	fig.update_xaxes(range=[0, 35.2])
	fig.update_layout(annotations=annotations, title="Trustworthiness") 
	return fig

def plotCoefSilValues(titles, silVecs):
	fig = go.Figure([go.Bar(x=titles, y=silVecs, marker_color='indianred')])
	fig.update_layout(title="Coeficiente Silhouette")
	return fig

def plotHistory(titles, vec_sil, vec_neigh, vec_f1score):
	dicts = []
	if len(vec_sil) != 0:
		dicts.append(dict(range = [-1,1], label = 'Coeficiente Silhouette', values = vec_sil))
	if len(vec_neigh) != 0:
		dicts.append(dict(range = [0,1], label = 'Trustworthiness (avg)', values = vec_neigh))
	if len(vec_f1score) != 0:
		dicts.append(dict(range = [0,1], label = 'f1-score', values = vec_f1score))

	fig = go.Figure(data=go.Parcoords(dimensions = list(dicts)))
	
	fig.update_traces(line_colorbar_tickvals=list(range(len(titles))), selector=dict(type='parcoords'))
	fig.update_traces(line_colorbar_ticktext=titles, selector=dict(type='parcoords'))
	fig.update_traces(line_color=list(range(len(titles))), selector=dict(type='parcoords'))
	return fig

