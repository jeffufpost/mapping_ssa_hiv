# Data
import pandas as pd
import numpy as np
import datetime
import math
from urllib.request import urlopen
import json
import requests
import urllib.request
import urllib.parse
import time
import io
from bs4 import BeautifulSoup
# Graphing
import plotly.graph_objects as go
import plotly.express as px
# Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
# Navbar
from navbar import Navbar

## Import necessary components
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import openpyxl

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from scipy.stats import zscore

#from app import create_time_series2, create_bar_series2
#from homepage import create_bar_series, create_time_series, create_prob_series

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])

app.config.suppress_callback_exceptions = True

server = app.server

##################################
#####   DATA    ##################
##################################
# With this:

## Import previously imputed.xlsx dataset
df=pd.read_excel('https://renkulab.io/gitlab/jeffrey.post/ssa_hiv_ml/-/raw/master/data/new/imputation_output/imputed.xlsx?inline=false', engine='openpyxl', index_col=0).set_index(['Country', 'Survey'])


# Create dictionary for sliding window
# Create groups that include 5 years of surveys, so 2002 will be [2000,20004], 2003 = [2001,2005] and so on..
# First surveys are 2000, so the first group is 2002 = [2000,2004]
# Last survey is 2018, so the last group is 2016 = [2014, 2018]
min_year = 2002
max_year = 2016
G={}
for start_year in range(min_year, max_year+1):
    #G[start_year]=df.loc[(df.index.get_level_values('Survey') >= start_year-2) & (df.index.get_level_values('Survey') < start_year+3)]
    # The line below removes duplicate country entries in the 5year groups, so say 2015 has Senegal four times, for 2017, 2016, 2015 and 2014, this only keeps Senegal 2017 which limits one countries bias
    G[start_year]=df.loc[(df.index.get_level_values('Survey') >= start_year-2) & (df.index.get_level_values('Survey') < start_year+3)].reset_index().drop_duplicates(subset='Country').set_index(['Country','Survey'])

## Create PCA

n=2
#n=3

pca = PCA(n_components = n)

# Use zscore normalization or not?
# Yes then use_zscore = True else False
use_zscore = False
#use_zscore = True

# Use this variable to select if you run PCA on last group of countries G[2016] or on all countries
#use_all = True
use_all = False

# Fit model with data from the last group
if use_all == False:
    if use_zscore == False:
        X2D_G4 = pca.fit(G[max_year].drop(columns=['iso','cow','GY']))
    elif use_zscore == True:
        X2D_G4 = pca.fit(G[max_year].drop(columns=['iso','cow','GY']).apply(zscore))
if use_all == True:
    if use_zscore == False:
        X2D_G4 = pca.fit(df.drop(columns=['iso','cow','GY']))
    elif use_zscore == True:
        X2D_G4 = pca.fit(df.drop(columns=['iso','cow','GY']).apply(zscore))

# Create a similar dict but composed of the PCs
PCG={}

if n == 2:
    components = ['PC-1', 'PC-2']
elif n == 3:
    components = ['PC-1', 'PC-2', 'PC-3']

for start_year in range(min_year,max_year+1):
    if use_zscore == False:
        PCG[start_year]=pd.DataFrame(pca.transform(G[start_year].drop(columns=['iso','cow','GY'])), columns=components, index=G[start_year].index)
    elif use_zscore == True:
        PCG[start_year]=pd.DataFrame(pca.transform(G[start_year].drop(columns=['iso','cow','GY']).apply(zscore)), columns=components, index=G[start_year].index)

#PCG[start_year]=pd.DataFrame(pca.transform(G[start_year].drop(columns=['iso','cow','GY'])), columns=components, index=G[start_year].index)
#PCG_norm[start_year]=pd.DataFrame(pca_norm.transform(G[start_year].drop(columns=['iso','cow','GY']).apply(zscore)), columns=components, index=G[start_year].index)
#PCG[start_year]=pd.DataFrame(pca_norm.transform(G[start_year].drop(columns=['iso','cow','GY']).apply(zscore)), columns=components, index=G[start_year].index)

# Calculate loadings if you want
loadings=pd.DataFrame(pca.components_.T*np.sqrt(pca.explained_variance_), columns=components, index=G[min_year].drop(columns=['iso','cow','GY']).columns)

def minmax(X):
    return (X - X.min()) / (X.max() - X.min())

norm_loadings = minmax(loadings)
#norm_loadings_true = minmax(loadings_True)


fake = loadings.copy()
fake.iloc[:,:] = 0
blabliblou = fake.append(np.abs(loadings))

# Plot loadings
#fig = px.scatter(norm_loadings.reset_index(), x='PC-1', y='PC-2', color='index', text='index')
fig = px.line(blabliblou.reset_index(), x='PC-1', y='PC-2', color='index', text='index')
fig.update_traces(textposition='top center')
#fig = px.scatter(norm_loadings, x='PC-1', y='PC-2',hover_name=loadings.index)
fig.update_layout(height=800)

# Get HIV incidence and prevalence into usable format from Aziza's original dataset

HIV_incidence_path = 'data/original_aziza/HIVIncidence_per1000_15_49.csv'
#HIV_incidence_path = '../../data/new/New HIV infections_HIV incidence per 1000 population - Adults (15-49)_Population All adults (15-49).csv'

HIV_prevalence_path = 'data/original_aziza/HIVPrevalence_15_49.csv'
#HIV_prevalence_path = '../../data/new/People living with HIV_HIV Prevalence - Adults (15-49)_Population Adults (15-49).csv'

def read_HIV(path):
    tmp = pd.read_csv(path)
    tmp = tmp.set_index('Country').filter(regex='[0-9][0-9][0-9][0-9]$',axis=1)
    return tmp.rename(index={'Congo, Rep.': 'Congo', 'Congo, Dem. Rep.':'Congo Democratic Republic'})
#    return tmp.rename(index={'Democratic Republic of the Congo':'Congo Democratic Republic'})

HIV = read_HIV(HIV_incidence_path)
HIV_prev = read_HIV(HIV_prevalence_path)

def addHIV(X, incprev):
    tmp=[]
    if incprev == 'incidence':
        HIV = read_HIV(HIV_incidence_path)
    elif incprev == 'prevalence':
        HIV = read_HIV(HIV_prevalence_path)
    for i in range(len(X)):
        year=X.reset_index().Survey[i]
        if (year > 2017):
            year=2017
        tmp.append(HIV.loc[X.reset_index().Country[i],str(year)])
    return pd.Series(tmp).astype(float)

for start_year in range(min_year,max_year+1):
    PCG[start_year]['HIV.incidence']=addHIV(PCG[start_year], 'incidence').values

for start_year in range(min_year,max_year+1):
    PCG[start_year]['HIV.prevalence']=addHIV(PCG[start_year], 'prevalence').values

# Number of clusters we want is 3 (from inertia basically)
k=3

kmeans = KMeans(n_clusters=k, random_state=0)

ccenters={}
clabels={}

start_year=2002

# Use n to use n PCs
# Here we only want 2D clustering
#kmeans.fit(PCG[start_year].iloc[:,:2])
kmeans.fit(PCG[start_year].iloc[:,:n])

for start_year in range(min_year,max_year+1):
    #print(start_year)
    # The line below allows us to initiliaze the cluster from the previous centroids - this avoid the jump at year 2009
    kmeans = KMeans(n_clusters=k, random_state=0, init=kmeans.cluster_centers_)
    #kmeans.fit(PCG[start_year].iloc[:,:n])
    kmeans.fit(PCG[start_year].iloc[:,:n])
    #print(kmeans.cluster_centers_)

    # Sort clusters by their sum of PC-1 and PC-2
    #idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))

    # It's easier to sort by PC-1 instead since that is more of mark of clusters:
    idx = np.argsort(kmeans.cluster_centers_[:,0])
    lut = np.zeros_like(idx)
    lut[idx] = np.arange(k)

    #print(lut[idx])
    #print(lut)
    #print(kmeans.labels_)
    #print(lut[kmeans.labels_])

    #print([kmeans.cluster_centers_[i] for i in idx])
    ccenters[start_year]=[kmeans.cluster_centers_[i] for i in idx]
    PCG[start_year]['Cluster']=lut[kmeans.labels_]



for start_year in range(min_year,max_year+1):
    PCG[start_year]['gy']=start_year

centroids={}
for start_year in reversed(range(min_year,max_year+1)):
    if n==2:
        centroids[start_year]=pd.DataFrame([['Centroid_0', start_year, ccenters[start_year][0][0], ccenters[start_year][0][1], 10, 10, 0, start_year],['Centroid_1', start_year, ccenters[start_year][1][0], ccenters[start_year][1][1], 10, 10, 1, start_year],['Centroid_2', start_year, ccenters[start_year][2][0], ccenters[start_year][2][1], 10, 10, 2, start_year]], columns=['Country', 'Survey', 'PC-1', 'PC-2', 'HIV.incidence', 'HIV.prevalence', 'Cluster', 'gy']).set_index(['Country','Survey'])
    elif n==3:
        centroids[start_year]=pd.DataFrame([['Centroid_0', start_year, ccenters[start_year][0][0], ccenters[start_year][0][1], ccenters[start_year][0][2], 10, 10, 0, start_year],['Centroid_1', start_year, ccenters[start_year][1][0], ccenters[start_year][1][1], ccenters[start_year][1][2], 10, 10, 1, start_year],['Centroid_2', start_year, ccenters[start_year][2][0], ccenters[start_year][2][1], ccenters[start_year][2][2], 10, 10, 2, start_year]], columns=['Country', 'Survey', 'PC-1', 'PC-2', 'PC-3', 'HIV.incidence', 'HIV.prevalence', 'Cluster', 'gy']).set_index(['Country','Survey'])

# Same but using different cluster number for centroids than actual clusters they represent (in order to change symbol)
#centroids={}
#for start_year in reversed(range(min_year,max_year+1)):#
    #if n==2:
#        centroids[start_year]=pd.DataFrame([['Centroid_0', start_year, ccenters[start_year][0][0], ccenters[start_year][0][1], 10, 10, 3, start_year],['Centroid_1', start_year, ccenters[start_year][1][0], ccenters[start_year][1][1], 10, 10, 4, start_year],['Centroid_2', start_year, ccenters[start_year][2][0], ccenters[start_year][2][1], 10, 10, 5, start_year]], columns=['Country', 'Survey', 'PC-1', 'PC-2', 'HIV.incidence', 'HIV.prevalence', 'Cluster', 'gy']).set_index(['Country','Survey'])
#    elif n==3:
#        centroids[start_year]=pd.DataFrame([['Centroid_0', start_year, ccenters[start_year][0][0], ccenters[start_year][0][1], ccenters[start_year][0][2], 10, 10, 0, start_year],['Centroid_1', start_year, ccenters[start_year][1][0], ccenters[start_year][1][1], ccenters[start_year][1][2], 10, 10, 1, start_year],['Centroid_2', start_year, ccenters[start_year][2][0], ccenters[start_year][2][1], ccenters[start_year][2][2], 10, 10, 2, start_year]], columns=['Country', 'Survey', 'PC-1', 'PC-2', 'PC-3', 'HIV.incidence', 'HIV.prevalence', 'Cluster', 'gy']).set_index(['Country','Survey'])

#pd.concat([PCG[2000], PCG[2001]])
data_sy_list = []
for i, start_year in enumerate(PCG):
    data_sy = PCG[start_year]
    data_sy_list.append(data_sy)
final_data_sy = pd.concat(data_sy_list)

#pd.concat([PCG[2000], PCG[2001]])
data_centroids_list = []
for i, start_year in enumerate(centroids):
    data_centroids = centroids[start_year]
    data_centroids_list.append(data_centroids)
final_data_centroids = pd.concat(data_centroids_list)

#final_data_centroids['Centroid'] = 0
#final_data_sy['Centroid'] = 1

tmp=pd.concat([final_data_sy, final_data_centroids])
tmp['Cluster']=tmp['Cluster'].astype(str)

## 3. Animation 2002-2016 - PCA 2D or 3D space and HIV boxplots

size='HIV.incidence'

# diff = 10%
diff = 1.1

map=px.scatter(
    #px.scatter_3d(
    tmp.reset_index().drop_duplicates(subset=['Country', 'gy']),
    #final_data_sy.reset_index().drop_duplicates(subset=['Country', 'gy']),
    x="PC-1",
    y="PC-2",
    #z="PC-3",
    animation_frame="gy",
    hover_name="Country",
    size=size,
    color="Cluster",
    #symbol="Centroid",
    category_orders={'Cluster':['0','1','2', '4', '5']},
    color_discrete_sequence=['#890000','#2a6b28','#4f5a90', '#000000', '#000000' , '#000000'],
    text="Country",
    size_max=40,
    range_x=[tmp['PC-1'].min()*diff,tmp['PC-1'].max()*diff],
    range_y=[tmp['PC-2'].min()*diff,tmp['PC-2'].max()*diff],
    #range_z=[tmp['PC-3'].min()-diff,tmp['PC-3'].max()+diff],
)

map.update_layout(title="Evolution of countries on 2D PCA Space<br>Size represents {}".format(size), height=800)

#fig.write_html('all_pca_2D.html', auto_open=True)



variable='HIV.incidence'

figbox = px.box(
    final_data_sy.reset_index(),
    x="Cluster",
    y=variable,
    points='all',
    #notched=True,
    animation_frame="gy",
    hover_name='Country',
    color='Cluster',
    category_orders={'Cluster':[0,1,2]},
    color_discrete_sequence=['#890000','#2a6b28','#4f5a90'],
    hover_data=['Survey'], range_y=[0,tmp[variable].max()+5]
)
#figbox.update_traces(quartilemethod="inclusive")
figbox.update_layout(title="Evolution of {} by cluster".format(variable), height=800)

#fig.write_html('HIV2015.html', auto_open=True)


##################################
##################################

nav = Navbar()

header_FR = html.Div(
    [
        dbc.Row(
            dbc.Col(
                html.H2(
                    'COVID-19 in France',
                    style= {
                        'textAlign': 'center',
                        "background": "lightblue"
                    }
                )
            )
        )
    ]
)

div_figbox = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='figbox',
                    figure=figbox,
                    #clickData={'points': [{'location': '01'}]}
                )
            )
        )
    ]
)

div_map = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='map',
                    figure=map,
                    #clickData={'points': [{'location': '01'}]}
                )
            )
        )
    ]
)

div_fig = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dcc.Graph(
                    id='fig',
                    figure=fig,
                    #clickData={'points': [{'location': '01'}]}
                )
            )
        )
    ]
)

def App():
    layoutapp = html.Div([
        nav,
        header_FR,
        div_fig
    ])
    return layoutapp

def Homepage():
    layouthp = html.Div([
        nav,
        header_FR,
        div_map,
        div_figbox
    ])
    return layouthp

app.layout = html.Div([
    dcc.Location(id = 'url', refresh = True),
    html.Div(id = 'page-content')
])

@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return App()
    else:
        return Homepage()

if __name__ == '__main__':
    app.run_server(debug=True)
