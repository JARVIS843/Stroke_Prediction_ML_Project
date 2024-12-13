[Stroke Prediction using Neural Network](https://github.com/JARVIS843/Stroke_Prediction_ML_Project)
=========================================
---
## About:
[This project](https://github.com/JARVIS843/Stroke_Prediction_ML_Project) aims to predict whether an elderly person is likely to get stroke based on eleven distinct features such as gender, age, and diseases etc. It trains binary classification neural network [models](#models) using [TensorFlow](https://www.tensorflow.org/), on Jupyter Notebook.

--- 
## Installation Instructions:
To install / clone this project onto your machine, you should:

Run the following command:
```
conda update conda
git clone https://github.com/JARVIS843/Stroke_Prediction_ML_Project.git
cd Stroke_Prediction_ML_Project
conda env create -f environment.yml
conda activate StrokePredictionML
```

Add the environment (StrokePredictionML) to Jupyter:
```
python -m ipykernel install --user --name=StrokePredictionML --display-name "Python (StrokePredictionML)"
```

(Optional): Confirm the kernel was added successfully:
```
jupyter kernelspec list
```
You would then need to manually select the kernel from the Jupyter Interface


---
## Models:

If you would like to use our pre-trained models, all of them can be found: [Here](./Models/)

Below is a table delineating the specifications and performances for each model:
| Models     | Accuracy|
| --------       | ------- |
| Logistic Regression        | 95.21%     |
| Random Forest        | 95.01%     |


---
## Dataset Used:

All of the models for this project are trained using the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

---
## Authors & Background:

This project is co-developed by: [Jarvis Yang](https://github.com/JARVIS843), and [Jegyeong An](https://github.com/northbear99).

The project was intended to be the final project for the [Introduction to Machine Learning](https://github.com/sdrangan/introml) course, provided by Professor [Sundeep Rangan](https://wireless.engineering.nyu.edu/sundeep-rangan/).

---
## License (MIT):

See License File
