# Official TensorFlow/Keras Implementation
**Property of:**
<p align="center">
  <img width="460" height="300" src="./sbp_logo.jpg">
</p>
 
 ![logo](https://img.shields.io/badge/tensorflow-2.0-green.svg?style=plastic)
 ![logo1](https://img.shields.io/badge/Keras-2.3.1-green.svg?style=plastic)
 ![logo2](https://img.shields.io/badge/CUDA-10.0-green.svg?style=plastic)
 ![logo2](https://img.shields.io/badge/cuDNN-7.6.4-green.svg?style=plastic)
 

**This repository contains the official TensorFlow/Keras implementation of the working paper: Analysis of lumber prices timeseries using long short-term memory artificial neural networks** 

For inquiries, please contact Dercilio Junior Verly Lopes at dvl23@msstate.edu



## System requirements

* Both Linux and Windows are supported, but we strongly recommend Linux for performance and compatibility reasons.
* 64-bit Python 3.7 installation. We recommend Anaconda 2.5 with newest numpy version.
* TensorFlow 2.0 and Keras 2.3.1 with GPU support.
* One or more high-end NVIDIA GPUs with at least 11GB of DRAM. We recommend NVIDIA RTX 2080 Ti GPU.
* NVIDIA driver 411.63, CUDA toolkit 10.0, cuDNN 7.6.4.

## Also tested on:

* TensorFlow 1.14 with Keras 2.2.5 with GPU support. Changes necessary in SBP/networks.py - line 24. (STRATEGY.SCOPE)
* Single NVIDIA RTX 2070 with 8GB OF DRAM. 
* NVIDIA driver 436.30 or newer, CUDA toolkit 9.0, cuDNN 7.6.2

## How to use?
Run **lstm.py within each folder**
