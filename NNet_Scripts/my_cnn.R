library(tensorflow)

# https://adeshpande3.github.io/adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html
# Combine structure from lacune cnn paper with learnings from ImageNet structures

# Same as lacune paper:
# AlexNet (2012):
# - ReLU nonlinearity found to decrease training time dramatically, in comparison to tanh
# - Data augmentation: data included reflections and translations of existing data
# - Stochastic gradient descent + fixed momentum and weight decay

# ZFNet (2013):
# - good paper on visualising conv filters
# - smaller convolution filter in the first couple of layers helps retain a lot of pixel information
# - later layers -> more filters
# - DeConvNet: undoes the conv net, so we can see what the resulting filters represent

# VGG Net (2014):
# - simple, with depth. 19 layers, all 3x3 filters, stride 1, pad 1; with 2x2 pooling, stride 2
# - smaller filters, with large numbers, were as effective as larger filters, but with fewer parameters. Can fit in more ReLU layers
# - Number of filters doubles after each pooling
# - Made new samples by changing scale

# GoogLeNet (2015):
# - Highly complex, inception module - many streams in parallel. Choice between 1x1 conv, 3x3 conv, 5x5 conv and 3x3 pooling at each layer.
# - Including 1x1 conv before the 3x3 and 5x5 reduces dimensionality, as well as allows for more ReLU.
# - Model is able to extract very fine detail, while having a large receptive field, all while being rather computational efficient.
# - 9 inception moduels - over 100 layers.
# - No fully connected layers, used average pool instead.
# - 12x fewer parameters than AlexNet.

# Microsoft ResNet (2015):
# - 152 layers!
# - Residual block: Input x. F(x) = conv-relu-conv. Then output x + F(x), so we're training F(x), a kind of residual.
# - Spatial size decreases from 224 to 56 after the first 2 layers

# R-CNN: region detecting models.
# - Particular regions of the image were highlighted that may contain an object of interest. The bounded area was reshaped and fed to AlexNet.
# - Fast R-CNN
# - Faster R-CNN
# - First detect regions where an object may be. Resize image before putting through convNet classifier.

# Hyper-parameter training:
# - Use a validation set! 50/25/25. 
# - Training set used to create model.
# - Apply different hyperparameters, and review performance based on validation set. Choose the model that performs best on the validation set.
# - Then test final model on the test set.



# Train the papers cnn first, to classify images.
# When that is done, try to implement RCNN, to determine candidate bounding boxes of any dimension, rather than the fixed 51x51 box.
# Train to find 4D output (x, y, width, height) of object. Use L2 distance loss between prediction and 'ground truth'.