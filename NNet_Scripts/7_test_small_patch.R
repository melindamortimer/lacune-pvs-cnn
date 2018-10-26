# Evaluation of slice. Outputs probabilities as heat map


# Import Model ------------------------------------------------------------

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

saver$restore(sess, "/srv/scratch/z5016924/correct_sampling_results/attempt2/model.ckpt")





# Eval --------------------------------------------------------------------


#load("/srv/scratch/z5016924/Data sets/attempt2/test_lacunes.Rda")
data.dir <- "/srv/scratch/z5016924/MAS_W2/"

id = "7663"

file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
soft <- f.read.nifti.volume(file.soft)

file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
t1 <- f.read.nifti.volume(file.t1)

file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
flair <- f.read.nifti.volume(file.flair)

file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
lacune <- f.read.nifti.volume(file.lacune)

max.rows <- 20000
data.image <- array(NA, dim = c(max.rows, 5206))
i <- 1

# for (x in 26:(dim(soft)[1]-26)) {
for (x in 50:150){
  y = 150
  # for (z in 26:(dim(soft)[3]-26)) {
  for (z in 40:140) {
  
    # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
    midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
    # Skip if pixel is not in brain matter, or is a lacune
    if (midregion == 0) next
    
    data.image[i, 1] <- as.numeric(id)
    data.image[i, 2] <- x
    data.image[i, 3] <- y
    data.image[i, 4] <- z
    
    patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
    patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
    
    data.image[i, 5:2605] <- patch.t1
    data.image[i, 2606:5206] <- patch.flair
    
    i <- i + 1
    if (i > max.rows) {
      stop(paste("Reached max number of rows", max.rows))
    }
  }
}
# Remove empties
data.image <- data.image[1:(i-1),]

