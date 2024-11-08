# -*- coding: utf-8 -*-
"""
====================================
Demo of HDBSCAN clustering algorithm
====================================
.. currentmodule:: sklearn

In this demo we will take a look at :class:`cluster.HDBSCAN` from the
perspective of generalizing the :class:`cluster.DBSCAN` algorithm.
We'll compare both algorithms on specific datasets. Finally we'll evaluate
HDBSCAN's sensitivity to certain hyperparameters.

We first define a couple utility functions for convenience.
"""
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option("display.precision", 2)

from sklearn.cluster import DBSCAN, HDBSCAN, KMeans
from sklearn.datasets import make_blobs
from statistics import mean
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA  


  # to plot the heat maps

df = pd.read_json("testing_datafile_AIGT.json", lines = True)
df_research = pd.read_json("testing_datafile_AIGT_research.json", lines = True)
df_research = df_research.drop(18)
df = df.drop(11)
df = pd.concat([df, df_research])


# select the numb columns and list columns
df2 = df[['distanceToTarget', 'timeBetweenKills', 'distanceBetweenKills']]
df = df[['accuracy','numberOfShots', 'numberOfKills', 'numberOfDeaths']]

# remove the empty one 

# get the mean of all lists
df2 = df2.map(mean)

# join both dfs
df3 = df.join(df2)

# Scale the data 
scaler = StandardScaler()
scaler.fit(df3)
df3_scaled = pd.DataFrame(scaler.transform(df3))
df3_scaled.columns = df3.columns 

# Show the heatmap for the correlation 
# print(sns.heatmap(df3_scaled.corr()))


# Transform the data to 2 dimensions
pca = PCA(n_components=2)
df_2dim = pd.DataFrame(pca.fit_transform(df3_scaled))



# Define the required functions
# Plot the scatterplot 
def plotting_scat(model, data = df):
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title(f"The model: {model} ")
    # in case the model has probabilites
    try:
        ser = model.probabilties_ 
    except:
        ser = None

    # use scatter plot with pca reduced 
    scatter = ax.scatter(df_2dim[0], df_2dim[1], c=model.labels_, s=ser)

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                loc="upper left", title="Classes")
    ax.add_artist(legend1)

    print(f"\n Scores of: {model}")
    print(f"Silhoute score of:", silhouette_score(data, model.labels_))
    print(f"Calinski and Harabasz score:", calinski_harabasz_score(data, model.labels_))
    print(f"Davies-Bouldin index ", davies_bouldin_score(data, model.labels_))
    plt.show()
def plotting_box(model):
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.set_title(f"The model: {model} ")

    # Melt the data for the plot
    df_melted = pd.melt(df3_scaled)
   
    # get all the labals reapeted
    lister = [model.labels_.tolist() * 11]
    final_list = []
    for listje in lister:
        final_list = final_list + listje

        
    final = df_melted.join(pd.DataFrame(final_list).rename(columns = {0:'label'}))
   
    # palette= {0: "purple", 1: "cornflowerblue", 2: "limegreen", 3 : "yellow"}

    sns.boxplot(data=final, x="variable", y = "value", hue = 'label')


    plt.show()

def gridsearch_db(parameter1, parameter2, dataframe = df, plot = False):
    best_score = -1
    best_model = None
    current = -10
    for first in parameter1:
        for min_s in parameter2:
            model = DBSCAN(eps= first, min_samples= min_s).fit(dataframe)
            # Gives errors when 1 label
            try:
                current = silhouette_score(dataframe, model.labels_)
                if plot:
                    plotting_scat(model, dataframe)
                    plotting_box(model)
            except:
                continue
            # if we want to qualitatively evaluate
            
            # Keep the best model
            if current > best_score: 
                best_score = current
                best_model = model
    return best_model

def gridsearch_km(parameter1, dataframe = df, plot = False):
    best_score = -1
    best_model = None
    current = -10

    for first in parameter1:
        model = KMeans(n_clusters=first, random_state=0, n_init="auto").fit(dataframe)
        # Gives errors when 1 label
        try:
            current = silhouette_score(dataframe, model.labels_)        
            # if we want to qualitatively evaluate
            if plot:
                print("hi")
                plotting_scat(model, dataframe)
                plotting_box(model)
        except:
            continue
        
        # Keep the best model
        if  current > best_score: 
            best_score = current
            best_model = model
    return best_model




# Train the models
# With gridsearch
epis = [2]
samples = [4]
n_clusters = [2, 3, 4, 5]

# hdbs = HDBSCAN().fit(df3_scaled)
# kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(df3_scaled)
# kmeans = gridsearch_km(n_clusters, df3_scaled, True)
dbs = gridsearch_db(epis, samples, df3_scaled, True)

# plot the models

# plotting_box(kmeans)
# plotting_scat(kmeans, df3_scaled)

# plotting_box(hdbs)
# plotting_scat(dbs, df3_scaled)










