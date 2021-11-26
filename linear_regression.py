import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

def linear_regression_model(fdx_train, fdx_test, fdy_train):
    # Develop two different models to classify CTG features into the three fetal health
    # states (I intentionally did not name which two models. Note that this is a multiclass
    # problem that can also be treated as regression, since the labels are numeric.) (2+2) 

    model = LinearRegression().fit(fdx_train, fdy_train)
    y_pred = model.predict(fdx_test)

    # Calculate Linear Regression Coefficients
    b1, b0 = np.polyfit(fdx_test["baseline value"], y_pred, 1)
    print(f"B1: {b1}, B0: {b0}")

    # Graph scatter plot and linear regression line
    plt.plot(fdx_test["baseline value"], b1 * fdx_test["baseline value"] + b0)
    plt.scatter(fdx_test["baseline value"], y_pred, color="pink")
    plt.xlabel("Basline")
    plt.ylabel("Fetal Health")
    plt.show()
