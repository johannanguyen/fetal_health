import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

def decision_tree_model(fdx_train, fdy_train):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 

    model_tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(fdx_train, fdy_train)
    plt.figure(figsize=(8, 6))
    tree.plot_tree(model_tree, fontsize=6)
    plt.show()
