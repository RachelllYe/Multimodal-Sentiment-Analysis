# Multimodal-Sentiment-Analysis
This repository is about the Final Project:Multimodal-Sentiment-Analysis of DaSE course *Contemporary Artificial Intelligence*

## Setup
This implemetation is based on Python3. To run the code, you need the following dependencies:

You can simply run<br>
```python
pip install -r requirements.txt
```
## Repository structure
```
|--data/ # includes all datas(images and text) of this experiment
|--TextEncoder.py # use pre-trained model BERT to encode the text
|--ImageEncoder.py # use pre-trainde model ResNet-152 to encode the image
|--model.py # concat the tensor of text and image
|--run.py # do training, validation and test
|--utils.py # includes the functions of data preprocessing and some other helper functions
|--configs.py # set the basic parameters
```

## Dataset
The Datasets includes 4000 pairs of image and text. We split training set and validation set according to 8:2. And the size of test set is 869 without label.
|Dataset|Size|
|:---|:---|
|train|3200|
|validation|800|
|test|869|

## Run pipeline on dataset
You can use <code>run.py</code> to run the experiment by simply passing the <code>model_type</code>, arguments. In total, we support three tasks(i.e.,['multimodal', 'unimodal_text', 'unimodal_img']). You can run them respectively to compare the performance. And more arguments you can set freely can be checked in the file <code>configs.py</code>.
|Tasks|Description|
|:---|:---|
|multimodal|the inputs include images and text|
|unimodal_text|the inputs just include text|
|unimodal_img|the inputs just include images|

For example, if you want to run multimodal with the default argument values:
```
python run.py --model_type multimodal
```
Besides, you can specify:
```
batch_size: set batch size
epochs: set the total epoch numbers of training
dropout: set the drop out rate to avoid overfitting
lr: set the learning rate
```
## Results
We use the arracy to measure model performance, and it is displayed as follows.
|Input|Accuracy|
|:---|:---|
|MultiModal|73.1%|
|Only Text|69.2%|
|Only Image|67%|
