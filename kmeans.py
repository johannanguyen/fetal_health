import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def kmeans(fetal_data, k):
    # Data frame with 2 columns: fetal_movement, uterine_contractions
    x = fetal_data.loc[:, 'fetal_movement':'uterine_contractions']
    print(x)

    kmeans = KMeans(k)
    kmeans.fit(x)

    clusters = kmeans.fit_predict(x)
    clustered_data = fetal_data.copy()
    clustered_data['Clusters'] = clusters 
    plt.scatter(clustered_data['fetal_movement'],clustered_data['uterine_contractions'],c=clustered_data['Clusters'],cmap='rainbow')
    plt.show()