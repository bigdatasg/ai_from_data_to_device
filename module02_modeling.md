# Module 2: Modeling with Tensorflow 2.0
## Overview
In the previous module, you have seen the power of tf.data APIs in TensorFlow 2.0 reading complex data types from any storage. 

In this module, you will see an example of data ETL from raw images to input into tensors, then apply transfer learning (which is how a lot of future models for end-users will be built at companies) to build an emotion classification model. We initially chose this use case so that at deployment in the next module, you can see the inference of your own facial expressions. However, there was some TF 2.0 compatibility issues that we have yet able to fix at deployment, but follows us to get the latest updates post workshop after we're able to fix the issues.

The work is adapted from the following [tutorial](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/images/transfer_learning.ipynb).

## What you'll learn
In this lab, you will learn to:
1. Examine and understand the data (not exhaustive)
2. Build an example input pipeline (there are many ways to build an input pipeline)
3. Compose your model:
    * Choose a suitable pre-train model
    * Choose the sub-model you will depend on to build your model
    * Append the suitable classification layers at the end
4. Train your model
5. Evaluate your model
6. Tune and/or update the architecture of your model

## Prerequisites

### Skills
- Python: able to manipulate lists, dictionaries, understand maps
- Proficiency with Jupyter
- High level understanding of TensorFlow APIs will be good
- Basic Machine Learning/Deep Learning theoretical understanding
- Basic statistical understanding of model metrics and how to compute them

### Equipment/Software

This setup is prepared for GCP resouces, and with no change, it could also be run on:
- Google Colab (you might need to check on dependencies for MS Azure Notebook and so on)
- Local Python notebook 

## Task 01: Transfer learning
Extract the embeddings learned by [MobileNetV2](https://ai.googleblog.com/2018/04/mobilenetv2-next-generation-of-on.html) pre-trained on [ImageNet](http://www.image-net.org/). You simply add a new classifier, which will be trained from scratch, on top of the pre-trained model so that you can repurpose the model for our dataset. For this workshop, we will be learning emotion classes with this [dataset](https://www.kaggle.com/aspiring1/fer2013-images).

## Task 02: Fine tuning
Unfreezing a few of the top layers of a frozen model base and jointly training both the newly-added classifier layers and the last layers of the base model. This allows us to "fine tune" the higher-order feature representations in the base model in order to make them more relevant for the specific task.

