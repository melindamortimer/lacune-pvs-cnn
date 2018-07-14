library(tensorflow)

# Data is now in 3D patches (instead of 2D) of scales 32x32x5, 64x64x5 and 128x128x5
# These 3 scales are all shaped into 32x32x5
# Each of the 3 scales goes through a separate network, structured the same way:

# - 6 conv layers
# - 1 pooling after 2nd conv
# - 1 fully connected layer of 300 neurons

# The 3x300 neurons are then concatenated
# Include 7 location features - gives a total neuron count of 907
# Location features describe:
# - x,y,z coords
# - distance to right ventricle, left ventricle, cortex and midsaggital brain surface

# The 907 were fully connected to 2 more fully connected layers, with 200 and 2 neurons
# Softmax

# He weight initialisation
he.init <- tf$contrib$layers$variance_scaling_initializer(factor = 2.0,
                                                          mode = "FAN_AVG",
                                                          uniform = FALSE)
weight.variable <- function(shape) {
  initial <- he.init(shape)
  tf$Variable(initial)
}

beta.variable <- function(shape) {
  tf$Variable(tf$zeros(shape))
}

scale.variable <- function(shape) {
  tf$Variable(tf$ones(shape))
}

conv3d <- function(x, W) {
  tf$nn$conv3d(x, W, strides = c(1L, 1L, 1L, 1L, 1L), padding = "SAME")
}

# Batch Normlisation
batch.norm <- function(z, beta, scale) {
  moments <- tf$nn$moments(z, 0L)
  tf$nn$batch_normalization(z, moments[[1]], moments[[2]]^2, beta, scale, 1e-3)
}

# Keep probability of 0.5
keep.prob <- tf$placeholder(tf$float32)

# Responses
y_ <- tf$placeholder(tf$float32, shape(NULL, 2L))

# 1st ConvNet. Input 32x32x5 -----------------------------------------------------------------------

l1.x <- tf$placeholder(tf$float32, shape(NULL, 32L*32L*5L))

# Reshape samples to 32x32x5, 2 channels (T1, FLAIR)
l1.x.image <- tf$reshape(x, shape(-1L, 32L, 32L, 5L, 2L))

# conv 1: 64 filters, 3x3x2 size
l1.W.conv1 <- weight.variable(shape(3L, 3L, 2L, 2L, 64L))
l1.z.conv1 <- conv3d(l1.x.image, l1.W.conv1)
# Batch normalisation
l1.beta.conv1 <- beta.variable(shape(64L))
l1.scale.conv1 <- scale.variable(shape(64L))
l1.bn.conv1 <- batch.norm(l1.z.conv1, l1.beta.conv1, l1.scale.conv1)
# ReLU activation
l1.h.conv1 <- tf$nn$relu(l1.bn.conv1)

# conv 2: 64 filters, 3x3x2 size
l1.W.conv2 <- weight.variable(shape(3L, 3L, 2L, 64L, 64L))
l1.z.conv2 <- conv3d(l1.h.conv1, l1.W.conv2)
l1.beta.conv2 <- beta.variable(shape(64L))
l1.scale.conv2 <- scale.variable(shape(64L))
l1.bn.conv2 <- batch.norm(l1.z.conv2, l1.beta.conv2, l1.scale.conv2)
l1.h.conv2 <- tf$nn$relu(l1.bn.conv2)

# pool 1: size 2x2x1
l1.h.pool1 <- tf$nn$max_pool3d(l1.h.conv2, ksize = c(1L, 2L, 2L, 1L, 1L),
                            strides = c(1L, 2L, 2L, 1L, 1L), padding = "VALID")

# conv 3: 128 filters, 3x3x1
l1.W.conv3 <- weight.variable(shape(3L, 3L, 1L, 64L, 128L))
l1.z.conv3 <- conv3d(l1.h.pool1, l1.W.conv3)
l1.beta.conv3 <- beta.variable(shape(128L))
l1.scale.conv3 <- scale.variable(shape(128L))
l1.bh.conv3 <- batch.norm(l1.z.conv3, l1.beta.conv3, l1.scale.conv3)
l1.h.conv3 <- tf$nn$relu(l1.bn.conv3)

# conv 4: 128 filters, 3x3x1
l1.W.conv4 <- weight.variable(shape(3L, 3L, 1L, 128L, 128L))
l1.z.conv4 <- conv3d(l1.h.conv3, l1.W.conv4)
l1.beta.conv4 <- beta.variable(shape(128L))
l1.scale.conv4 <- scale.variable(shape(128L))
l1.bh.conv4 <- batch.norm(l1.z.conv4, l1.beta.conv4, l1.scale.conv4)
l1.h.conv4 <- tf$nn$relu(l1.bn.conv4)

# conv 5: 256 filters, 3x3x1
l1.W.conv5 <- weight.variable(shape(3L, 3L, 1L, 128L, 256L))
l1.z.conv5 <- conv3d(l1.h.conv4, l1.W.conv5)
l1.beta.conv5 <- beta.variable(shape(256L))
l1.scale.conv5 <- scale.variable(shape(256L))
l1.bh.conv5 <- batch.norm(l1.z.conv5, l1.beta.conv5, l1.scale.conv5)
l1.h.conv5 <- tf$nn$relu(l1.bn.conv5)

# conv 6: 256 filters, 3x3x1
l1.W.conv6 <- weight.variable(shape(3L, 3L, 1L, 256L, 256L))
l1.z.conv6 <- conv3d(l1.h.conv5, l1.W.conv6)
l1.beta.conv6 <- beta.variable(shape(256L))
l1.scale.conv6 <- scale.variable(shape(256L))
l1.bh.conv6 <- batch.norm(l1.z.conv6, l1.beta.conv6, l1.scale.conv6)
l1.h.conv6 <- tf$nn$relu(l1.bn.conv6)

l1.h.conv6.flat <- tf$reshape(l1.h.conv6, shape(-1L, 6L*6L*3L*256L))

# full 1: 300 neurons
l1.W.fcl1 <- weight.variable(shape(6L*6L*3L*256L, 300L))
l1.z.fcl1 <- tf$matmul(l1.h.conv6.flat, W.fcl1)
l1.beta.fcl1 <- beta.variable(shape(300L))
l1.scale.fcl1 <- scale.variable(shape(300L))
l1.bn.fcl1 <- batch.norm(l1.z.fcl1, l1.beta.fcl1, l1.scale.fcl1)
l1.h.fcl1 <- tf$nn$relu(l1.bn.fcl1)
l1.h.fcl1.drop <- tf$nn$dropout(l1.h.fcl1, keep_prob = keep.prob)



# 2nd ConvNet. Input 64x64x5 -----------------------------------------------------------------------

# Reduce down to 32x32x5

l2.x <- tf$placeholder(tf$float32, shape(NULL, 32L*32L*5L))

# Reshape samples to 32x32x5, 2 channels (T1, FLAIR)
l2.x.image <- tf$reshape(x, shape(-1L, 32L, 32L, 5L, 2L))

# conv 1: 64 filters, 3x3x2 size
l2.W.conv1 <- weight.variable(shape(3L, 3L, 2L, 2L, 64L))
l2.z.conv1 <- conv3d(l2.x.image, l2.W.conv1)
# Batch normalisation
l2.beta.conv1 <- beta.variable(shape(64L))
l2.scale.conv1 <- scale.variable(shape(64L))
l2.bn.conv1 <- batch.norm(l2.z.conv1, l2.beta.conv1, l2.scale.conv1)
# ReLU activation
l2.h.conv1 <- tf$nn$relu(l2.bn.conv1)

# conv 2: 64 filters, 3x3x2 size
l2.W.conv2 <- weight.variable(shape(3L, 3L, 2L, 64L, 64L))
l2.z.conv2 <- conv3d(l2.h.conv1, l2.W.conv2)
l2.beta.conv2 <- beta.variable(shape(64L))
l2.scale.conv2 <- scale.variable(shape(64L))
l2.bn.conv2 <- batch.norm(l2.z.conv2, l2.beta.conv2, l2.scale.conv2)
l2.h.conv2 <- tf$nn$relu(l2.bn.conv2)

# pool 1: size 2x2x1
l2.h.pool1 <- tf$nn$max_pool3d(l2.h.conv2, ksize = c(1L, 2L, 2L, 1L, 1L),
                               strides = c(1L, 2L, 2L, 1L, 1L), padding = "VALID")

# conv 3: 128 filters, 3x3x1
l2.W.conv3 <- weight.variable(shape(3L, 3L, 1L, 64L, 128L))
l2.z.conv3 <- conv3d(l2.h.pool1, l2.W.conv3)
l2.beta.conv3 <- beta.variable(shape(128L))
l2.scale.conv3 <- scale.variable(shape(128L))
l2.bh.conv3 <- batch.norm(l2.z.conv3, l2.beta.conv3, l2.scale.conv3)
l2.h.conv3 <- tf$nn$relu(l2.bn.conv3)

# conv 4: 128 filters, 3x3x1
l2.W.conv4 <- weight.variable(shape(3L, 3L, 1L, 128L, 128L))
l2.z.conv4 <- conv3d(l2.h.conv3, l2.W.conv4)
l2.beta.conv4 <- beta.variable(shape(128L))
l2.scale.conv4 <- scale.variable(shape(128L))
l2.bh.conv4 <- batch.norm(l2.z.conv4, l2.beta.conv4, l2.scale.conv4)
l2.h.conv4 <- tf$nn$relu(l2.bn.conv4)

# conv 5: 256 filters, 3x3x1
l2.W.conv5 <- weight.variable(shape(3L, 3L, 1L, 128L, 256L))
l2.z.conv5 <- conv3d(l2.h.conv4, l2.W.conv5)
l2.beta.conv5 <- beta.variable(shape(256L))
l2.scale.conv5 <- scale.variable(shape(256L))
l2.bh.conv5 <- batch.norm(l2.z.conv5, l2.beta.conv5, l2.scale.conv5)
l2.h.conv5 <- tf$nn$relu(l2.bn.conv5)

# conv 6: 256 filters, 3x3x1
l2.W.conv6 <- weight.variable(shape(3L, 3L, 1L, 256L, 256L))
l2.z.conv6 <- conv3d(l2.h.conv5, l2.W.conv6)
l2.beta.conv6 <- beta.variable(shape(256L))
l2.scale.conv6 <- scale.variable(shape(256L))
l2.bh.conv6 <- batch.norm(l2.z.conv6, l2.beta.conv6, l2.scale.conv6)
l2.h.conv6 <- tf$nn$relu(l2.bn.conv6)

l2.h.conv6.flat <- tf$reshape(l2.h.conv6, shape(-1L, 6L*6L*3L*256L))

# full 1: 300 neurons
l2.W.fcl1 <- weight.variable(shape(6L*6L*3L*256L, 300L))
l2.z.fcl1 <- tf$matmul(l2.h.conv6.flat, W.fcl1)
l2.beta.fcl1 <- beta.variable(shape(300L))
l2.scale.fcl1 <- scale.variable(shape(300L))
l2.bn.fcl1 <- batch.norm(l2.z.fcl1, l2.beta.fcl1, l2.scale.fcl1)
l2.h.fcl1 <- tf$nn$relu(l2.bn.fcl1)
l2.h.fcl1.drop <- tf$nn$dropout(l2.h.fcl1, keep_prob = keep.prob)





# 3rd ConvNet. Input 128x128x5 -----------------------------------------------------------------------

# Reduce down to 32x32x5

l3.x <- tf$placeholder(tf$float32, shape(NULL, 32L*32L*5L))

# Reshape samples to 32x32x5, 2 channels (T1, FLAIR)
l3.x.image <- tf$reshape(x, shape(-1L, 32L, 32L, 5L, 2L))

# conv 1: 64 filters, 3x3x2 size
l3.W.conv1 <- weight.variable(shape(3L, 3L, 2L, 2L, 64L))
l3.z.conv1 <- conv3d(l3.x.image, l3.W.conv1)
# Batch normalisation
l3.beta.conv1 <- beta.variable(shape(64L))
l3.scale.conv1 <- scale.variable(shape(64L))
l3.bn.conv1 <- batch.norm(l3.z.conv1, l3.beta.conv1, l3.scale.conv1)
# ReLU activation
l3.h.conv1 <- tf$nn$relu(l3.bn.conv1)

# conv 2: 64 filters, 3x3x2 size
l3.W.conv2 <- weight.variable(shape(3L, 3L, 2L, 64L, 64L))
l3.z.conv2 <- conv3d(l3.h.conv1, l3.W.conv2)
l3.beta.conv2 <- beta.variable(shape(64L))
l3.scale.conv2 <- scale.variable(shape(64L))
l3.bn.conv2 <- batch.norm(l3.z.conv2, l3.beta.conv2, l3.scale.conv2)
l3.h.conv2 <- tf$nn$relu(l3.bn.conv2)

# pool 1: size 2x2x1
l3.h.pool1 <- tf$nn$max_pool3d(l3.h.conv2, ksize = c(1L, 2L, 2L, 1L, 1L),
                               strides = c(1L, 2L, 2L, 1L, 1L), padding = "VALID")

# conv 3: 128 filters, 3x3x1
l3.W.conv3 <- weight.variable(shape(3L, 3L, 1L, 64L, 128L))
l3.z.conv3 <- conv3d(l3.h.pool1, l3.W.conv3)
l3.beta.conv3 <- beta.variable(shape(128L))
l3.scale.conv3 <- scale.variable(shape(128L))
l3.bh.conv3 <- batch.norm(l3.z.conv3, l3.beta.conv3, l3.scale.conv3)
l3.h.conv3 <- tf$nn$relu(l3.bn.conv3)

# conv 4: 128 filters, 3x3x1
l3.W.conv4 <- weight.variable(shape(3L, 3L, 1L, 128L, 128L))
l3.z.conv4 <- conv3d(l3.h.conv3, l3.W.conv4)
l3.beta.conv4 <- beta.variable(shape(128L))
l3.scale.conv4 <- scale.variable(shape(128L))
l3.bh.conv4 <- batch.norm(l3.z.conv4, l3.beta.conv4, l3.scale.conv4)
l3.h.conv4 <- tf$nn$relu(l3.bn.conv4)

# conv 5: 256 filters, 3x3x1
l3.W.conv5 <- weight.variable(shape(3L, 3L, 1L, 128L, 256L))
l3.z.conv5 <- conv3d(l3.h.conv4, l3.W.conv5)
l3.beta.conv5 <- beta.variable(shape(256L))
l3.scale.conv5 <- scale.variable(shape(256L))
l3.bh.conv5 <- batch.norm(l3.z.conv5, l3.beta.conv5, l3.scale.conv5)
l3.h.conv5 <- tf$nn$relu(l3.bn.conv5)

# conv 6: 256 filters, 3x3x1
l3.W.conv6 <- weight.variable(shape(3L, 3L, 1L, 256L, 256L))
l3.z.conv6 <- conv3d(l3.h.conv5, l3.W.conv6)
l3.beta.conv6 <- beta.variable(shape(256L))
l3.scale.conv6 <- scale.variable(shape(256L))
l3.bh.conv6 <- batch.norm(l3.z.conv6, l3.beta.conv6, l3.scale.conv6)
l3.h.conv6 <- tf$nn$relu(l3.bn.conv6)

l3.h.conv6.flat <- tf$reshape(l3.h.conv6, shape(-1L, 6L*6L*3L*256L))

# full 1: 300 neurons
l3.W.fcl1 <- weight.variable(shape(6L*6L*3L*256L, 300L))
l3.z.fcl1 <- tf$matmul(l3.h.conv6.flat, W.fcl1)
l3.beta.fcl1 <- beta.variable(shape(300L))
l3.scale.fcl1 <- scale.variable(shape(300L))
l3.bn.fcl1 <- batch.norm(l3.z.fcl1, l3.beta.fcl1, l3.scale.fcl1)
l3.h.fcl1 <- tf$nn$relu(l3.bn.fcl1)
l3.h.fcl1.drop <- tf$nn$dropout(l3.h.fcl1, keep_prob = keep.prob)






# Concatenated + Fully Connected -------------------------------------------------------------

# Location features vector length 7
# Location data: x,y,z + 4 distances
location.features <- tf$placeholder(tf$float32, shape(NULL, 7L))

# Concatenated (fully connected #1 - 907 neurons)
h.fcl1 <- tf$concat(list(l1.h.fcl1.drop, l2.h.fcl1.drop, l3.h.fcl1.drop, location.features),
                     axis = 0L)

# full 2: 200 neurons
W.fcl2 <- weight.variable(shape(907L, 200L))
z.fcl2 <- tf$matmul(h.fcl1, W.fcl2)
beta.fcl2 <- beta.variable(shape(200L))
scale.fcl2 <- scale.variable(shape(200L))
bn.fcl2 <- batch.norm(z.fcl2, beta.fcl2, scale.fcl2)
h.fcl2 <- tf$nn$relu(bn.fcl2)
h.fcl2.drop <- tf$nn$dropout(h.fcl2, keep_prob = keep.prob)

# full 3: 2 neurons
W.fcl3 <- weight.variable(shape(200L, 2L))
z.fcl3 <- tf$matmul(h.fcl2.drop, W.fcl3)
beta.fcl3 <- beta.variable(shape(2L))
scale.fcl3 <- scale.variable(shape(2L))
bn.fcl3 <- batch.norm(z.fcl3, beta.fcl3, scale.fcl3)

# softmax
y <- tf$nn$softmax(bn.fcl3)



# Training ----------------------------------------------------------------

# Cross entropy loss
# L2 regularisation (Ridge Regression) with lambda_2 = 2e-5
cross.entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y), reduction_indices = 1L))
l2.reg <- cross.entropy + 2e-5 * tf$reduce_sum(W.fcl3^2)

# Decaying learning rate from 5e-4, decay factor 2 when training accuracy drops.
learn.rate <- tf$Variable(5e-4)

# Adam updater
train.step <- tf$train$AdamOptimizer(learn.rate)$minimize(l2.reg)

correct.prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct.prediction, tf$float32))
best.accuracy <- tf$Variable(0.0)
prev.accuracy <- tf$Variable(0.0)

sess <- tf$Session()

sess$run(tf$global_variables_initializer())

# Early stopping
saver <- tf$train$Saver(ls(pattern = '^l[123].(W|beta|scale)'))

# Training for 40 epochs
max.epochs <- 40
# Decaying learning rate. 5e-4 reduced to 1e-6
for (e in 1:max.epochs) {
  # Randomise data
  # while (next.batch of 128) {
    # train.step$run(feed_dict = dict(
    #   x = batch[[1]], y_ = batch[[2]], keep.prob = 0.5))
  # }
  if (e %% 10) {
    train.accuracy <- accuracy$eval(feed_dict = dict(
      x = batch[[1]], y_ = batch[[2]], keep.prob = 1.0))
    cat(sprintf("step %d, training accuracy %g\n", e, train.accuracy))
  }
  # Model selection by highest accuracy on validation set
  if(sess$run(train.accuracy) > sess$run(best.accuracy)) {
    sess$run(tf$assign(best.accuracy, train.accuracy))
    saver$save(sess, './1_lacune_cnn_2/model', global_step = e)
  }
  # Learning rate halves when accuracy decreases
  if(sess$run(train.accuracy) < sess$run(prev.accuracy)) {
    sess$run(tf$assign(learn.rate, learn.rate/2))
  }
  sess$run(tf$assign(prev.accuracy, train.accuracy))

}
saver$restore(sess, tf$train$latest_checkpoint('./1_lacune_cnn_1/'))


# test.accuracy <- accuracy$eval(feed_dict = dict(
#   x = test.set.images, y_ = test.set.labels, keep.prob = 1.0, learning.rate = learning.rates[n]
# ))
# cat(sprintf("test accuracy %g\n", test.accuracy))

sess$close()


# Then parameters adjusted to optimise accuracy on validation set (hyper-parameters: network depth, mini-batch size, initial learning rate, decaying factor, lambda_2, dropout rate)