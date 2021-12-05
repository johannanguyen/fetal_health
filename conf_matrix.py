import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

def conf_matrix(fdy_test, y_pred, model_name):

    cm = confusion_matrix(fdy_test, y_pred)
    cm_df = pd.DataFrame(cm,
                     index = ['Normal','Suspect','Pathological'], 
                     columns = ['Normal','Suspect','Pathological'])
                     
    #Plotting the confusion matrix
    plt.figure(figsize=(7.5,7.5))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title(model_name+' CONFUSION MATRIX')
    plt.ylabel('ACTUAL VALUES')
    plt.xlabel('PREDICTED VALUES')
    plt.show()