import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans(fetal_data, k):
    x = fetal_data.loc[:, 'baseline value':'fetal_movement']
    print(x)

    kmeans = KMeans(k)
    kmeans.fit(x)

    clusters = kmeans.fit_predict(x)
    clustered_data = fetal_data.copy()
    clustered_data['Clusters'] = clusters 
    plt.scatter(clustered_data['baseline value'],clustered_data['fetal_movement'],c=clustered_data['Clusters'],cmap='rainbow')
    plt.title("K = " + str(k))
    plt.ylabel('fetal_movement')
    plt.xlabel('baseline value')
    plt.show()