# Models:
---

## About:
Congrats, you have located our model folder. 
Below is a full list of the pre-trained models with their respective accuracies ranked:
* Extra Trees - 99%
* Gradient Boost - 98%
* Random Forest - 98% 
* XG Boost - 98%
* Logistic Regression (Deprecated) - 95.21%
* Nadam Neural Network (Best) - 90%
  
*Note: The Logistic Regression and Random Forest model inside the [Models](/Models/) folder is developed with a different preprocessing technique, and is thus deprecated. The Extra Trees, Gradient Boost, new Random Forest, and XG Boost model would soon be uploaded here if I have time.

### Neural Network:
The trained best 6 layered (5 hidden 1 output) neural network model has the following architecture:

| Layer (type)                    | Output Shape           | Param #       |
|---------------------------------|------------------------|---------------|
| Hidden_Layer_1 (Dense)          | (None, 256)            | 5,120         |
| batch_normalization_265         | (None, 256)            | 1,024         |
| (BatchNormalization)            |                        |               |
| activation_265 (Activation)     | (None, 256)            | 0             |
| dropout_265 (Dropout)           | (None, 256)            | 0             |
| Hidden_Layer_2 (Dense)          | (None, 128)            | 32,896        |
| batch_normalization_266         | (None, 128)            | 512           |
| (BatchNormalization)            |                        |               |
| activation_266 (Activation)     | (None, 128)            | 0             |
| dropout_266 (Dropout)           | (None, 128)            | 0             |
| Hidden_Layer_3 (Dense)          | (None, 64)             | 8,256         |
| batch_normalization_267         | (None, 64)             | 256           |
| (BatchNormalization)            |                        |               |
| activation_267 (Activation)     | (None, 64)             | 0             |
| dropout_267 (Dropout)           | (None, 64)             | 0             |
| Hidden_Layer_4 (Dense)          | (None, 32)             | 2,080         |
| batch_normalization_268         | (None, 32)             | 128           |
| (BatchNormalization)            |                        |               |
| activation_268 (Activation)     | (None, 32)             | 0             |
| dropout_268 (Dropout)           | (None, 32)             | 0             |
| Hidden_Layer_5 (Dense)          | (None, 16)             | 528           |
| batch_normalization_269         | (None, 16)             | 64            |
| (BatchNormalization)            |                        |               |
| activation_269 (Activation)     | (None, 16)             | 0             |
| dropout_269 (Dropout)           | (None, 16)             | 0             |
| Output_Layer (Dense)            | (None, 1)              | 17            |

It's worth mentioning that the model uses [Swish](https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/framework/activations/Swish) (swish(x) = x * sigmoid(x)) for activation functions of the hidden units, and the model was trained on Nadam optimizer, 0.01 Learning Rate, 64 Batch Size, and 50 Epochs. Detailed graphic analysis of this model can be found in the Accuracy Analysis section of [Neural_Network_Model.ipynb](/Neural_Network_Model.ipynb).

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