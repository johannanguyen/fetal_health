import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans(fetal_data, k):
    # Data frame with 2 columns: baseline value, accelerations
    x = fetal_data.iloc[:,:2]
    print(x)

    kmeans = KMeans(k)
    kmeans.fit(x)

    clusters = kmeans.fit_predict(x)
    clustered_data = fetal_data.copy()
    clustered_data['Clusters'] = clusters 
    plt.scatter(clustered_data['baseline value'],clustered_data['accelerations'],c=clustered_data['Clusters'],cmap='rainbow')
    plt.show()
