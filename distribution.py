from helper_functions import *

def distribution(fetal_data):
    # Present a visual distribution of the 3 classes. Is the data balanced?
    # How do you plan to circumvent the data imbalance problem, if there is one?
    # (Hint: stratification needs to be included.) (1)
    baseline_list = fetal_data["baseline value"].tolist()
    health_list = fetal_data["fetal_health"].tolist()
    length = len(baseline_list)
    baseline_health = []

    for i in range(0, length):
        baseline_health.append( [baseline_list[i], health_list[i]] )

    class_one = divide_classes(baseline_health, 1.0)
    class_two = divide_classes(baseline_health, 2.0)
    class_three = divide_classes(baseline_health, 3.0)

    histogram(class_one)
    histogram(class_two)
    histogram(class_three)