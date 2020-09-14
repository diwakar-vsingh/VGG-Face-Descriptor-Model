# VGG Face Descriptor Model
Model Conversion: Convert the VGG face descriptor model to TensorFlow format using TensorFlow 2.x API.

----------------
Overview
----------------

Framework: TensorFlow 2.x

The easiest way is to upload the VGG_Gender.ipynb notebook on Google Colab and run it.

In the first cell, I have given a flag named _GPU. If it is True then change the runtime type to GPU else change the runtime type to None.

The following steps are taken:
1. The base model is created using the VGG model architecture from paper.
2. The base model weights are loaded using vgg_face.mat
3. The base model is created from the VGG model with the last layer removed.
4. The base model is frozen to avoid destroying any of the information they contain during future training rounds.
5. A new trainable layer is added on top of the frozen layers. This will be used to turn the old features into predictions on a new dataset.
6. The new model is trained on the provided dataset.
7. Final model architecture and weights are saved in HDF5 format (VGG_GENDER.h5)
8. The model is evaluated on a test image.

The whole dataset (aligned and valid) is split into two parts: training (85%) and validation set (15%)

Loss function: Binary Cross-Entropy 
Metric: Binary Accuracy

When _GPU = False, I get around 96.75% accuracy on the validation set. [EPOCH = 1, No added layer on top of VGG]
When _GPU = True, I get accuracy between 97% - 98% on validation set. [EPOCH = 3, 2 dense layer added on top of VGG]

If I would have been allowed to use GPU, I could even unfreeze some of the top layers of the base model and fine-tune the model further to reach an accuracy of around 99%.

------------------
References
------------------

To read vgg_face.mat file into my network, I referred to this article:
https://sefiks.com/2019/07/15/how-to-convert-matlab-models-to-keras/

Also, I made significant use of official TensorFlow tutorials and guides.
