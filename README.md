# news_article_summarizer_using_custom_model

## Overview

A NLP based summarizer which can summarize news article. Using T5TokenizerFast I have trained a custom  model on t5-base architecture and the dataset was collected from kaggle. 



## Table of Contents
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Data Set](#data-set)
- [Data Modelling](#data-modelling)
- [Inference](#inference)
- [HuggingFace Deployment](#huggingface-deployment)
- [FlaskApp](#flaskapp)



## Installation

To get started with this project, follow these steps:

Clone the repository:
   ```bash
   git clone https://github.com/marufc36/news_article_summarizer_using_custom_model
   ```
  
## Dependencies
   ```bash
pip3 install requirements.txt
```




## Data set

Data-set is collected from [Kaggle](https://www.kaggle.com/datasets/sunnysai12345/news-summary). The dataset consists of 4515 examples and contains Author_name, Headlines, Url of Article, Short text, Complete Article. I have only worked with Short text and Complete Article and other columns were dropped from the data-set. 

## Data Modelling 

Using t5-base pretrained model, T5TokenizerFast, pytorch lighting the model was trained to three epoch on the collected data-set.Short text was used as summary and complete article was used as text. 


Colab Notebook
   ```bash
https://drive.google.com/file/d/1qbCqMXjv5aqyfkJ6ntQMfGwYoXXATqWM/view?usp=sharing
```

## Inference

I have inferenced the using a gradio app. I have to load the best checkpoint using NewsSummaryModel. I have created model.py module which includes NewsSummaryModel.


## HuggingFace Deployment
The model was deployed to HuggingFace Spaces. Linkn is given below.

   ```bash
https://huggingface.co/spaces/mmchowdhury/News_Summary_With_Custom_Model
```
![image](https://github.com/marufc36/news_article_summarizer_using_custom_model/assets/151602012/e0bf7d6b-0d1b-4272-96eb-038c7996fb28)



## FlaskApp
Using HuggingFace Api I have build a flask app. 
![image](https://github.com/marufc36/news_article_summarizer_using_custom_model/assets/151602012/78784412-05d0-4c10-893c-d672f2d10dd1)

