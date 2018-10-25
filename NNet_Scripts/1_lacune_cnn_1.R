library(tensorflow)
library(crayon)
# rm(list = ls())
load("/srv/scratch/z5016924/Data sets/attempt2/training.Rda")
load("/srv/scratch/z5016924/Data sets/attempt2/validation.Rda")
load("/srv/scratch/z5016924/Data sets/attempt2/testing.Rda")


# 1 -----------------------------------------------------------------------

# Each sample was a subimage around a candidate lacune. Each sample is actually a 51x51 patch, from both T1 and FLAIR (2 channels)

# 7 layers:
# - 4 conv
# - 1 pool
# - 3 fully connected

# Data placeholders
# Unlimited samples, 51x51 = 2601. 2 channels
x <- tf$placeholder(tf$float32, shape(NULL, 5202L))
# Unlimited samples, 2 outcomes, lacune or not
y_ <- tf$placeholder(tf$float32, shape(NULL, 2L))

# Weights initialised by He Method (maybe can test Xavier init?)
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

conv2d <- function(x, W) {
  tf$nn$conv2d(x, W, strides = c(1, 1, 1, 1), padding = "VALID")
}

# Batch Normlisation
batch.norm <- function(z, beta, scale) {
  moments <- tf$nn$moments(z, 0L)
  tf$nn$batch_normalization(z, moments[[1]], moments[[2]]^2, beta, scale, 1e-3)
}


# Dropout of 0.3 on fully connected layers (avoids overfitting)
# SAME: new.height = ceil(old.height/stride)
# VALID: new.height = ceil((old.height - filter.height + 1)/stride)
keep.prob <- tf$placeholder(tf$float32)


# ReLU applied to neurons during conv (prevents vanishing gradient problem)

# Reshape x into however many samples, of 51x51, 2 colour channels (FLAIR & T1)
x.image <- tf$reshape(x, shape(-1L, 51L, 51L, 2L))


# Conv Layers -------------------------------------------------------------

# conv 1: 20 filters, 7x7 size
W.conv1 <- weight.variable(shape(7L, 7L, 2L, 20L))
z.conv1 <- conv2d(x.image, W.conv1)
# Batch normalisation
beta.conv1 <- beta.variable(shape(20L))
scale.conv1 <- scale.variable(shape(20L))
bn.conv1 <- batch.norm(z.conv1, beta.conv1, scale.conv1)
h.conv1 <- tf$nn$relu(bn.conv1)

# pool: 2x2 size, 2 stride
h.pool1 <- tf$nn$max_pool(h.conv1, ksize = c(1L,2L,2L,1L),
                          strides = c(1L,2L,2L,1L), padding = "VALID")

# conv 2: 40 filters, 5x5 size
W.conv2 <- weight.variable(shape(5L, 5L, 20L, 40L))
z.conv2 <- conv2d(h.pool1, W.conv2)
beta.conv2 <- beta.variable(shape(40L))
scale.conv2 <- scale.variable(shape(40L))
bn.conv2 <- batch.norm(z.conv2, beta.conv2, scale.conv2)
h.conv2 <- tf$nn$relu(bn.conv2)

# conv 3: 80 filters, 3x3 size
W.conv3 <- weight.variable(shape(3L, 3L, 40L, 80L))
z.conv3 <- conv2d(h.conv2, W.conv3)
beta.conv3 <- beta.variable(shape(80L))
scale.conv3 <- scale.variable(shape(80L))
bn.conv3 <- batch.norm(z.conv3, beta.conv3, scale.conv3)
h.conv3 <- tf$nn$relu(bn.conv3)

# conv 4: 110 filters, 3x3 size
W.conv4 <- weight.variable(shape(3L, 3L, 80L, 110L))
z.conv4 <- conv2d(h.conv3, W.conv4)
beta.conv4 <- beta.variable(shape(110L))
scale.conv4 <- scale.variable(shape(110L))
bn.conv4 <- batch.norm(z.conv4, beta.conv4, scale.conv4)
h.conv4 <- tf$nn$relu(bn.conv4)


# Fully Connected Layers --------------------------------------------------

h.conv4.flat <- tf$reshape(h.conv4, shape(-1L, 14L*14L*110L))
# full 1: 300 size
W.fcl1 <- weight.variable(shape(14L*14L*110L, 300L))
z.fcl1 <- tf$matmul(h.conv4.flat, W.fcl1)
beta.fcl1 <- beta.variable(shape(300L))
scale.fcl1 <- scale.variable(shape(300L))
bn.fcl1 <- batch.norm(z.fcl1, beta.fcl1, scale.fcl1)
h.fcl1 <- tf$nn$relu(bn.fcl1)
h.fcl1.drop <- tf$nn$dropout(h.fcl1, keep.prob)

# full 2: 200 size
W.fcl2 <- weight.variable(shape(300L, 200L))
z.fcl2 <- tf$matmul(h.fcl1.drop, W.fcl2)
beta.fcl2 <- beta.variable(shape(200L))
scale.fcl2 <- scale.variable(shape(200L))
bn.fcl2 <- batch.norm(z.fcl2, beta.fcl2, scale.fcl2)
h.fcl2 <- tf$nn$relu(bn.fcl2)
h.fcl2.drop <- tf$nn$dropout(h.fcl2, keep.prob)

# full 3: 2 size
W.fcl3 <- weight.variable(shape(200L, 2L))
z.fcl3 <- tf$matmul(h.fcl2.drop, W.fcl3)
beta.fcl3 <- beta.variable(shape(2L))
scale.fcl3 <- scale.variable(shape(2L))
bn.fcl3 <- batch.norm(z.fcl3, beta.fcl3, scale.fcl3)

# Softmax classifier
y <- tf$nn$softmax(bn.fcl3)


# Training ----------------------------------------------------------------

# Cross entropy loss
# L2 regularisation (Ridge Regression) with lambda_2 = 0.0001
# Note L1 regularisation is Lasso Regression
cross.entropy <- tf$reduce_mean(-tf$reduce_sum(y_ * tf$log(y + 1e-10), reduction_indices = 1L))
l2.reg <- cross.entropy + 0.0001 * tf$reduce_sum(W.fcl3^2)

# Stochastic gradient descent (Adam update)
learn.rate <- tf$placeholder(tf$float32)
train.step <- tf$train$AdamOptimizer(learn.rate)$minimize(l2.reg)

correct.prediction <- tf$equal(tf$argmax(y, 1L), tf$argmax(y_, 1L))
accuracy <- tf$reduce_mean(tf$cast(correct.prediction, tf$float32))

saver <- tf$train$Saver()

sess <- tf$InteractiveSession()

sess$run(tf$global_variables_initializer())

# Early stopping

# Current format not in epochs
# Randomise samples, then take batches of 128
num.samples <- nrow(training)
max.epochs <- 40
# Decaying learning rate. 5e-4 reduced to 1e-6
learning.rates <- seq(5e-4, 1e-6, length.out = max.epochs)

train.accuracy <- numeric(max.epochs*num.samples)
i.train.acc <- 1
train.accuracy2 <- numeric(max.epochs)
i.train.acc2 <- 1


# training ----------------------------------------------------------------



best.accuracy <- 0
e <- 1
start.timer <- proc.time()
while (e < max.epochs) {
  # randomise data
  stoch.training <- training[sample(nrow(training)),]
  for (i in seq(1, num.samples-128, by = 128)) {
    cat(white$bgBlack(paste("Epoch:",e)))
    cat(paste(" Analysing sample: ",i, "/",num.samples, " = ",round(i/num.samples,3)*100,"%", sep = ""))
    cat(paste(" Elapsed:", round((proc.time() - start.timer)[[3]]/60,2), "min"))
    train.step$run(feed_dict = dict(
      x = stoch.training[i:(i+127), 5:5206], y_ = stoch.training[i:(i+127), 5207:5208], keep.prob = 0.7, learn.rate = learning.rates[e]))
    
    # Report training accuracy
    if (i %% 5 == 0) {
      train.accuracy[i.train.acc] <- accuracy$eval(feed_dict = dict(
        x = training[i:(i+127), 5:5206], y_ = training[i:(i+127), 5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
      cat(sprintf(" Acc: %g", train.accuracy[i.train.acc]))
      i.train.acc <- i.train.acc + 1
      plot(train.accuracy[max(i.train.acc-500,1):(i.train.acc - 1)], type = "b")
    }
    cat("\n")
  }
  # Reporting testing accuracy
  n.valid <- 3000
    train.accuracy2[i.train.acc2] <- accuracy$eval(feed_dict = dict(
      x = validation[1:n.valid,5:5206], y_ = validation[1:n.valid,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
    cat(sprintf("epoch %d, testing accuracy %g\n", e, train.accuracy2[i.train.acc2]))
    
    # MANUAL ACCURACY TESTING
    # accuracy$eval(feed_dict = dict(x = testing[,5:5206], y_ = testing[,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
    
    
  # Early stopping - highest accuracy on validation set
  if(train.accuracy2[i.train.acc2] > best.accuracy) {
    cat("Saving Model..\n")
    best.accuracy <- train.accuracy2[i.train.acc2]
    saver$save(sess, "/srv/scratch/z5016924/correct_sampling_results/attempt2/model.ckpt")
  }
    i.train.acc2 <- i.train.acc2 + 1
    
  e <- e + 1
}



# Batch Accuracy Testing --------------------------------------------------
# When data was split into train/test only, and positives only made up 7% of data

# 85% of the way through epoch 12, accuracy sudden plummets from around 100%, down to near 0??
# test.up.to <- nrow(testing)
# test.up.to <- 1000
# accuracy$eval(feed_dict = dict(x = testing[1:test.up.to,5:5206], y_ = testing[1:test.up.to,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
# # Testing up to 5000 samples at that cut-off training gives accuracy of 99.74%
# 
# # Check all testing accuracy (in batches since it can't evaluate all at once)
# num.testing <- dim(testing)[1]
# testing.seq <- seq(1, num.testing, by = 5000)
# testing.accuracy <- numeric(length(testing.seq))
# for (i in 1:length(testing.seq)) {
#   print(paste(i,"of", length(testing.seq)))
#   testing.accuracy[i] <- accuracy$eval(feed_dict = dict(x = testing[i:min(i+4999, num.testing),5:5206], y_ = testing[i:min(i+4999, num.testing),5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
# }
# mean(testing.accuracy)
# # 0.9938



# Testing Set -------------------------------------------------------------
# Positives now make 1/3 of the data. Data split into training/validation/testing

# accuracy$eval(feed_dict = dict(x = testing[,5:5206], y_ = testing[,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))



saver$restore(sess, "/srv/scratch/z5016924/correct_sampling_results/attempt2/model.ckpt")
# saver$restore(sess, tf$train$latest_checkpoint("/srv/scratch/z5016924/model1/attempt5"))

sess$close()

# Cut out 0s
nrow.train.acc <- max(which(train.accuracy != 0))
train.accuracy <- train.accuracy[1:nrow.train.acc]

# Training accuracy
save(train.accuracy, file = "/srv/scratch/z5016924/correct_sampling_results/attempt2/train_accuracy.Rda")

# Epoch validation accuracy
save(train.accuracy2, file = "/srv/scratch/z5016924/correct_sampling_results/attempt2/train_accuracy2.Rda")



# Converting Full to Convolutional ----------------------------------------

# https://tech.hbc.com/2016-05-18-fully-connected-to-convolutional-conversion.html



# Evaluation --------------------------------------------------------------


truepos <- tf$reduce_sum(tf$cast(tf$logical_and(tf$equal(tf$argmax(y, 1L), 0L), tf$equal(tf$argmax(y_, 1L), 0L)), tf$float32))
trueneg <- tf$reduce_sum(tf$cast(tf$logical_and(tf$equal(tf$argmax(y, 1L), 1L), tf$equal(tf$argmax(y_, 1L), 1L)), tf$float32))
falsepos <- tf$reduce_sum(tf$cast(tf$logical_and(tf$equal(tf$argmax(y, 1L), 0L), tf$equal(tf$argmax(y_, 1L), 1L)), tf$float32))
falseneg <- tf$reduce_sum(tf$cast(tf$logical_and(tf$equal(tf$argmax(y, 1L), 1L), tf$equal(tf$argmax(y_, 1L), 0L)), tf$float32))

# ntest <- dim(testing)[1]
# truepos$eval(feed_dict = dict(x = testing[1:ntest,5:5206], y_ = testing[1:ntest,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
# trueneg$eval(feed_dict = dict(x = testing[1:ntest,5:5206], y_ = testing[1:ntest,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
# falsepos$eval(feed_dict = dict(x = testing[1:ntest,5:5206], y_ = testing[1:ntest,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
# falseneg$eval(feed_dict = dict(x = testing[1:ntest,5:5206], y_ = testing[1:ntest,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))


# Batch True/False Pos/neg evaluation
tfpn.eval <- numeric(4)
testseq <- seq(1,nrow(testing), by = 1000)

names(tfpn.eval) <- c("TP","TN","FP","FN")

for (i in testseq) {
  print(paste(i, "of", nrow(testing)))
  lower <- i
  upper <- min(i+999, nrow(testing))
  tfpn.eval[1] <- tfpn.eval[1] + truepos$eval(feed_dict = dict(x = testing[lower:upper,5:5206], y_ = testing[lower:upper,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
  tfpn.eval[2] <- tfpn.eval[2] + trueneg$eval(feed_dict = dict(x = testing[lower:upper,5:5206], y_ = testing[lower:upper,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
  tfpn.eval[3] <- tfpn.eval[3] + falsepos$eval(feed_dict = dict(x = testing[lower:upper,5:5206], y_ = testing[lower:upper,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
  tfpn.eval[4] <- tfpn.eval[4] + falseneg$eval(feed_dict = dict(x = testing[lower:upper,5:5206], y_ = testing[lower:upper,5207:5208], keep.prob = 1.0, learn.rate = learning.rates[e]))
}

tfpn.eval
round(tfpn.eval/nrow(testing),4)


# Search for true positive samples etc

tp.test <- tf$logical_and(tf$equal(tf$argmax(y, 1L), 0L), tf$equal(tf$argmax(y_, 1L), 0L))
tn.test <- tf$logical_and(tf$equal(tf$argmax(y, 1L), 1L), tf$equal(tf$argmax(y_, 1L), 1L))
fp.test <- tf$logical_and(tf$equal(tf$argmax(y, 1L), 0L), tf$equal(tf$argmax(y_, 1L), 1L))
fn.test <- tf$logical_and(tf$equal(tf$argmax(y, 1L), 1L), tf$equal(tf$argmax(y_, 1L), 0L))

found <- 0
for (i in 1:1000) {
  found <- tn.test$eval(feed_dict = dict(x = array(testing[i, 5:5206], dim = c(1, 5202)), y_ = array(testing[i, 5207:5208], dim = c(1, 2)), keep.prob = 1.0, learn.rate = learning.rates[1]))
  
  if (found) break
}

lower <- 3630
upper <- 3640
# Number of TRUE in range
falseneg$eval(feed_dict = dict(x = testing[lower:upper, 5:5206], y_ = testing[lower:upper, 5207:5208], keep.prob = 1.0, learn.rate = learning.rates[1]))

# Showing TRUE and FALSE values in range
fn.test$eval(feed_dict = dict(x = testing[lower:upper, 5:5206], y_ = testing[lower:upper, 5207:5208], keep.prob = 1.0, learn.rate = learning.rates[1]))

y$eval(feed_dict = dict(x = testing[1:100, 5:5206], y_ = testing[1:100, 5207:5208], keep.prob = 1.0, learn.rate = learning.rates[1]))


for(tmp in 1:100) {
  print(y$eval(feed_dict = dict(x = array(testing[tmp, 5:5206], dim = c(1, 5202)), y_ = array(testing[tmp, 5207:5208], dim = c(1,2)), keep.prob = 1.0, learn.rate = learning.rates[1])))
  readline(prompt = "enter")
}

