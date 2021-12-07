from helper_functions import *
import pandas as pd

def distribution(fetal_data):
    # Present a visual distribution of the 3 classes. Is the data balanced?
    # How do you plan to circumvent the data imbalance problem, if there is one?
    # (Hint: stratification needs to be included.) (1)
    health = fetal_data["fetal_health"]
    class_freq = [0, 0, 0]

    for item in health:
        class_freq[int(item)-1] += 1

    health = pd.DataFrame({"classifier": [1, 2, 3], "frequency": class_freq})
    health.plot.bar(x='classifier', y='frequency', rot=0)
    plt.show()