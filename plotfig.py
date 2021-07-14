import plotly
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


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