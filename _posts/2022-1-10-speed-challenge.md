---
layout: post
title: Speed challenge
---

The goal of this project was to predict speed of the car from video stream
and also learn how to write neural networks and datasets in Tensorflow.

This post describes my thought process for predicting speed of a driving 
car from video stream. The solution provided was to [Comma.ai speed challenge](https://github.com/commaai/speedchallenge). 
Provided dataset is one `train.mp4` video with color stream and `train.txt` text file
with speed of the car for corresponding frame in video. First, I investigated provided video
to see what the images look like. All source code is provided on [https://github.com/styczen/speed-challenge](https://github.com/styczen/speed-challenge). Below is sample 
frame provided in dataset for the challenge.

![](/images/speed-challenge/full_sample_frame.png)

Additionaly I read random frames from the whole video stream and visualize them with
corresponding speed:

![](/images/speed-challenge/random_batch.png)

## Preprocessing

As you can see there is a lot of unnecesary information like [vignetting](https://en.wikipedia.org/wiki/Vignetting) 
on the edges of the images which is brightness reduction closer to the periphery.
Another part is also car hood on the bottom of each frame. This part of view is always
stationary relative to camera, so there is not a lot of information which can be
useful. However, there are some reflections of the scene in the hood but it is
distorted and it could only make it harder to automatically extract features. 

So I decided to crop each image. Below is the scene from the image above but cropped
on all sides.

![](/images/speed-challenge/cropped_sample_frame.png)

Script used to load images from MP4 file, cropped and saved to images is in 
[`prepare_data.py`](https://github.com/styczen/speed-challenge/blob/master/prepare_data.py) file.

Next, I had to decide what approach might be the best for predicting speed of a car.
One valid solution might be to use sparse [optical flow](https://en.wikipedia.org/wiki/Optical_flow) 
which can be used to calculate motion of objects. In particular measuring motion of
some characteristic points. One of the most common algorithms to calculate optical
flow is [Lucas-Kanade method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method).

## Deep learning

During the time when I was looking at this challenge, I was learning about Tensorflow from
Google which is a free and open-source library which is used to train and run inference
of neural networks. In general operations performed on *tensors* which are generalization
of scalars and vectors. With CUDA programming language, those operations can be
greatly accelerated. 

I thought that it could be a great idea to learn Tensorflow, how to create specific
objects, dataset and train neural network model.

Tensorflow allows user to define object which inherits from `tf.keras.utils.Sequence`
for which user has to implement `__getitem__` and `__len__` methods. First one returns
input data, in our case images of some kind, in *batches* which are multiple images.
Later they are run through the network at once.

## Input data format

I knew that I cannot just load image and expect network to predict speed from single frame. 
Network is not able to learn anything substantial from static frame. There might be some blurs
on car edges because of movement but it is not enough. One option would be to build neural network
with recurrent elements like [`LSTM`](https://en.wikipedia.org/wiki/Long_short-term_memory).
At this time, my knowledge was not very great to start this approach but I was not sure about 
dataset size. It was only a single video and even with some augmentation e.g. mirroring,
flipping the convolutional neural network with recurrent elements would not be able to learn.

I felt that it was not a good approach, that you have to think about some smarter way of 
soving this problem. I thought how to include information about movement in single frame
which could be feed to simple CNN (convolutional neural network). How about reading 3 
consecutive frames, converting them from color to grayscale and stacking them as separate 
channels creating kind of RGB image? In this weird image is information about movement
encoded in channels. Sample stacked images:

![](/images/speed-challenge/input_stacked_channels.png)

Different colors on edges can be seen on first and last image can be seen. On the second one,
there are no additional colors because car is not moving. Additionaly on the third image, 
color shifts are quite small because car has just started moving.

## Neural network

Next step was to design neural network which can be used to predict speed from newly 
designed stacked 3 frames. To create your own model you have multiple options using
Tensorflow:

- use `tf.keras.Sequential` API,
- use *Functional API* like `tf.keras.Input`, `tf.keras.layers.Dense` API.

The main difference between two options is that using the first one, user can only build 
models which are stacked layers on top of the other. With *Functional API* user has much 
more flexibility with model architecture. Each layer can have multiple inputs and outputs,
you can branch out inside model. But if you want to build sequential model you can also 
do this with *Functional API*. I decided to sequential API because I knew that I will 
not need more flexibility to modify my model. Here is the code which builds this model:

```python
def get_model(input_shape, params={}):
    """Get convolutional model."""
    model = Sequential(name=params.get('name', 'Speed_predictor'))
    # Conv 0
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(2, 2), padding='same', 
                               input_shape=input_shape))
    model.add(ELU())

    # Conv 1
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(ELU())

    # Conv 2
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    model.add(ELU())

    # Conv 3
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ELU())

    # Conv 4
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(ELU())

    # Flat feature maps for FC layers
    model.add(Flatten())

    # FC 5
    model.add(Dense(64))
    model.add(ELU())

    # FC 6
    model.add(Dense(32))
    model.add(ELU())

    # # FC 7
    # model.add(Dense(16))
    # model.add(ELU())

    # FC 8
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer=params.get('optimizer', Adam()), 
                  loss=params.get('loss', 'mse'), 
                  metrics=params.get('metrics', ['mae']))
    return model 
```

Firstly, I stacked 5 convolutional layers with `ELU` (Exponential Linear Unit) 
activation function. Most of the time users use `ReLu` (Rectified Linear Units) but there 
are some differences between both:

- `ELU` becomes smooth slowly whereas `ReLu` zeros sharply when input is 0 or less,
- `ReLu` can make some gradients die because you have no information what happened to neuron when it is negative.

For those reasons and because `ELU` is a strong alternative to `ReLu` I decided to use those 
as activation function.



## Results

In progress

## Conclusions

Calculate optical flow and calculate speed from it or feed it to CNN.
