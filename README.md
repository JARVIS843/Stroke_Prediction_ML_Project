[Stroke Prediction using Machine Learning](https://github.com/JARVIS843/Stroke_Prediction_ML_Project)
=========================================
---
## About:
[This project](https://github.com/JARVIS843/Stroke_Prediction_ML_Project) aims to predict whether people across all age groups is likely to get strokes, based on eleven distinct features such as gender, age, and diseases etc. It trains various binary classification  [models](#models) using different methods, on Jupyter Notebook.

--- 
## Installation Instructions:

### Setup Conda Environment
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

### Setup Tensorflow with CUDA (Only for [Neural Network Models](#neural-network-models))
It took me 2 hours to setup everything up correctly, so I decided to put the instructions here.

Since this project relies on tensorflow 2.18.0, so **Windows Native WOULD NOT WORK!!!** (TF gave up its development since 2.11). However, WSL still does.

Before everything, you need to make sure you have installed the newest driver. If you are using Nvidia graphics card that supports CUDA 12.5, you need to make sure your driver version is at least 555.42.02 for Linux (NOT WSL), and 555.85 for Windows. You may manually download specific driver on Nvidia's website, but I recommend using [Nvidia Geforce Experience](https://www.nvidia.com/en-us/geforce/geforce-experience/download/). If you are using WSL2, then you only have to install the newest driver on Windows side, and **ABSOLUTELY, DO NOT INSTALL IT ON WSL LINUX**, as it will mess up everything.

Then, according to [this](https://www.tensorflow.org/install/source?hl=en#gpu), install [CUDA 12.5](https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Linux). Note, if you are using WLS2 Ubuntu, choose [WSL-Ubuntu](https://developer.nvidia.com/cuda-12-5-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0) in the Distribution section (as the Ubuntu version includes driver and may mess up the driver installed on Windows). 
After installation, check your installation with:
```
nvcc --version
```
If it's not found, then you have to add it to PATH with:
```
export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
```


Finally, you need to install [cuDNN 9.3](https://developer.nvidia.com/cudnn-9-3-0-download-archive?target_os=Linux) and follow the instruction on the webpage.

To verify that you have done everything correctly, use the following code (not command) in your jupyter notebook, with the previously established environment and kernel (StrokePredictionML). If setup correctly, it should not be zero (unless your GPU does not support CUDA 12.5 to begin with)
```
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```

And you should be done. If you ever encounter any problems, please consult the following links, as they helped me a bunch when I'm doing this myself:
* https://www.tensorflow.org/install/source?hl=en#gpu
* https://www.tensorflow.org/install/pip#linux
* https://www.tensorflow.org/install/pip#linux
* https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/ (Windows, but again, Windows Native doesn't work)
* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html (Linux)
* https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html


---
## Project Structure:
The project is divided into two parts: [Non-nerual-network models](#non-neural-network-models), and [Neural Network models](#neural-network-models)

### Non-Neural-Network Models:
This part of the project is in [ML_project.ipynb](/ML_project.ipynb), and it's responsible for data clean up, data analysis, data preprocessing, and training a variety of models (Extra Trees Classifier, Gradient boosting, Random Forest, and XGboost). The accuracies of the models are displayed with visualized confusion matrices.

### Neural Network Models:
This is the other part of the project, and can be found in [Neural_Network_Model.ipynb](/Neural_Network_Model.ipynb). In order to be consistent, it uses the same data clean up and preprocessing procedures. It employs a [tensorflow](https://www.tensorflow.org/) 6 layered neural network (5 hidden 1 output and specific specs can be found in [Models](/Models/)), and cross tests its accuracies with various optimizers (RMSprop, Nadam,and Adam), learning rates (0.01, 0.001,and 0.0001), batch sizes (32 ,and 64), as well as number of epochs (50 ,and 100). Adapted Learning Rate strategies are also attempted.





---

## Models:

If you would like to use our pre-trained models, or to see the performances of them, all of them can be found: [Here](./Models/)



***Note:** The models are serialized and exported using [Pickle](https://docs.python.org/3/library/pickle.html)

---
## Dataset Used:

All of the models for this project are trained using the [Kaggle Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). 

In the beginning of [ML_project.ipynb](./ML_project.ipynb), you are provided an option to use either the pre-downloaded dataset (downloaded on 12/13/2024) in [Dataset](./Dataset/), or to download the newest dataset from Kaggle if you are using Colab or Linux.

---
## Authors & Background:

This project is co-developed by: [Jarvis Yang](https://github.com/JARVIS843) (responsible for [Neural Network Model](#neural-network-models)), and [Jegyeong An](https://github.com/northbear99) (responsible for [Non-Neural-Network Models](#non-neural-network-models)).

The project was intended to be the final project for the [Introduction to Machine Learning](https://github.com/sdrangan/introml) course, provided by Professor [Sundeep Rangan](https://wireless.engineering.nyu.edu/sundeep-rangan/).

---
## License (MIT):

See License File
