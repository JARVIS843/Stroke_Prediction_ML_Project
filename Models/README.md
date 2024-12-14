# Models:
---

## About:
Congrats, you have located our model folder. 
Below is a full list of the pre-trained models with their respective accuracies:
* Logistic Regression - 95.21% 
* Random Forest - 95.01% 
* Neural Network - N/A
---

## How to Use:
The models are serialized and saved using [Pickle](https://docs.python.org/3/library/pickle.html). Hence, to use the pre-trained models, you must deserialize them first. Below is an example loading and using the logistic regression model:
```

# Load the logistic regression model from the Models folder
with open(Models/logistic_regression_model.pkl, 'rb') as f:
    lr_loaded = pickle.load(f)

# Use the loaded model for predictions
y_pred_loaded = lr_loaded.predict(X_test)

```