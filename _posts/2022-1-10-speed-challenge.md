---
layout: post
title: Speed challenge
---

The goal of this project was to predict speed of the car from video stream
and also learn how to write neural networks and datasets in Tensorflow.

This post describes my thought process for predicting speed of a driving 
car from video stream. The idea for the project came from [Comma.ai speed challenge](https://github.com/commaai/speedchallenge). 
Provided dataset is one `train.mp4` video with color stream and `train.txt` text file
with speed of the car for corresponding frame in video. All source code is provided on [https://github.com/styczen/speed-challenge](https://github.com/styczen/speed-challenge).

First, I investigated provided video to see what the images look like. 
Below is sample frame provided in the dataset for the challenge.

![](/images/speed-challenge/full_sample_frame.png){: .align-center}

Additionaly I read random frames from the whole video stream and visualize them with
corresponding speed:

![](/images/speed-challenge/random_batch.png){: .align-center}

## Preprocessing

As you can see there is a lot of unnecesary information like [vignetting](https://en.wikipedia.org/wiki/Vignetting) 
on the edges of the images which is brightness reduction closer to the periphery.
Another part is also car hood on the bottom of each frame. This part of view is always
stationary relative to camera, so there is not a lot of information which can be
useful. However, there are some reflections of the scene in the hood but it is
distorted and it could only make it harder to automatically extract useful features. 

So I decided to crop each image. Below is the scene from the image above but cropped
on all sides.

![](/images/speed-challenge/cropped_sample_frame.png){: .align-center}

Script used to load images from MP4 file, crop and save each frame to 
files is in [`prepare_data.py`](https://github.com/styczen/speed-challenge/blob/master/prepare_data.py) file.

Next, I had to decide what approach might be the best for predicting speed of a car.
One valid solution might be to use sparse [optical flow](https://en.wikipedia.org/wiki/Optical_flow) 
which can be used to calculate motion of objects. In particular measuring motion of
some characteristic points. One of the most common algorithms to calculate optical
flow is [Lucas-Kanade method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method).
More on that subject in future post.

## Deep learning

During the time when I was looking at this challenge, I was learning about Tensorflow from
Google which is a free and open-source library which is used to train and run inference
of neural networks. In general operations performed on *tensors* which are generalization
of scalars and vectors. With CUDA programming language, those operations can be
greatly accelerated on GPU. 

I thought that it could be a great idea to learn Tensorflow, how to create specific
objects, dataset and train neural network model.

Tensorflow allows user to define object which inherits from `tf.keras.utils.Sequence`
for which `__getitem__` and `__len__` methods have to implemented. First one returns
input data, in our case images of some kind, in *batches* which are multiple images.
The latter function returns number of batches which is number of all images divided 
by size of training batch. Later each batch is run through the network at once.

## Input data

I knew that I cannot just load image and expect network to predict speed from single frame. 
Network is not able to learn anything substantial from static frame. 
There might be some blurs on car edges because of movement but 
it is not enough. One option would be to build neural network with recurrent 
elements like [`LSTM`](https://en.wikipedia.org/wiki/Long_short-term_memory).
At this time, my knowledge was not very great to start this approach 
and I was not sure about dataset size. 
It was only a single video and even with some augmentation e.g. mirroring,
flipping, the convolutional neural network with recurrent elements would not be able to learn.

I felt that it was not a good approach (with this amount of data), 
that you have to think about some smarter way of 
soving this problem. I thought how to include information about movement in single frame
which could be feed to simple CNN (convolutional neural network). 
I came up with the idea to read 3 consecutive frames, convert them 
from color to grayscale and stack them as separate channels, 
creating kind of RGB image In this weird image there is information about movement
encoded in channels. Sample stacked images:

![](/images/speed-challenge/input_stacked_channels.png){: .align-center}

Different colors on edges can be seen on first and last image can be seen. On the second one,
there are no additional colors because car is not moving. Additionaly on the third image, 
color shifts are quite small because car has just started moving.

Next, loaded dataset was split in 0.6 - 0.2 - 0.2 parts. This means that whole *training*
part of dataset is 60% of the whole data and *validation* and *test* parts 
are both 20% each. Validation part most of the time is used to 
fine tune hyperparameters e.g. convolutional filter sizes, 
strides, number of layer, number of neurons.

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
do this with *Functional API*. I decided to use sequential API because I knew that I will 
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

For those reasons I decided to use `ELU` as activation function just to test it. 
Nevertheless `ReLu` activation is one the most used function of all, because it has
very simple computational complexity and derivative is very simple.

Additionally summary of the created model:

```
Model: "Speed_predictor"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 120, 160, 16)      1216      
_________________________________________________________________
elu (ELU)                    (None, 120, 160, 16)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 60, 80, 32)        12832     
_________________________________________________________________
elu_1 (ELU)                  (None, 60, 80, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 40, 64)        51264     
_________________________________________________________________
elu_2 (ELU)                  (None, 30, 40, 64)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 30, 40, 128)       73856     
_________________________________________________________________
elu_3 (ELU)                  (None, 30, 40, 128)       0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 30, 40, 128)       147584    
_________________________________________________________________
elu_4 (ELU)                  (None, 30, 40, 128)       0         
_________________________________________________________________
flatten (Flatten)            (None, 153600)            0         
_________________________________________________________________
dense (Dense)                (None, 64)                9830464   
_________________________________________________________________
elu_5 (ELU)                  (None, 64)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080      
_________________________________________________________________
elu_6 (ELU)                  (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33        
=================================================================
Total params: 10,119,329
Trainable params: 10,119,329
Non-trainable params: 0
_________________________________________________________________
```

The loss functions used was `MSE` (mean squared error) which was also used
to evaluate my delivarable.

## Results

First training run was done on dataset as it is, i.e. not shuffled or transformed at all.
I suspected that the model might overfit. One sign of this is when training loss decreases
but validation loss increases, the model overfits. The model then just learned the dataset.

Tensorflow allows to log different metrics, not only loss function value. 
For this training run, I decided to log `MAE` (mean absolute error). 

Below there are plots with training error (blue plots) and validation errors
(orange plots)

![](/images/speed-challenge/losses_first_run_overfit.png){: .align-center}

Model overfits badly. One simple solution to eliminate this problem is shuffling 
the dataset. Model then is not able to learn order of images and it converges faster.

And the new plots after shuffling:

![](/images/speed-challenge/losses_second_run.png){: .align-center}

Model doesn't overfit more, so shuffling of dataset was good to eliminate this problem.
Then I evaluated trained model on test split of dataset with results:

`Loss: 0.21525, MAE: 0.34983`

So result is not the best possible because MSE is around **0.21** and on the challenge
Github page is information that error lower then 10 is good result. Probably if I had more
time to test other ideas, I could get the error to lower values.

## Conclusions

Here are some conclusions and ideas after finishing this project:

- calculate optical flow and calculate speed from it or feed it to CNN,
- too big learning rate e.g. 0.001 resulted in instability in learning - after 
  a few epochs error erupted to a big value,
- shuffling data before learning is a very important step,
- using some CNN with recurrent elements might improve results but much 
  more data would be needed,
- using cv2.VideoCapture is too slow to use it when training - use only 
  to display results on full resolution images.
