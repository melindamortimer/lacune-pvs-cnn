# Apply trained model to entire scan and observe candidate lacunes.

library(AnalyzeFMRI)
library(tensorflow)

# Import Model ------------------------------------------------------------

x <- tf$placeholder(tf$float32, shape(NULL, 5202L))
y_ <- tf$placeholder(tf$float32, shape(NULL, 2L))

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

batch.norm <- function(z, beta, scale) {
  moments <- tf$nn$moments(z, 0L)
  tf$nn$batch_normalization(z, moments[[1]], moments[[2]]^2, beta, scale, 1e-3)
}

keep.prob <- tf$placeholder(tf$float32)

x.image <- tf$reshape(x, shape(-1L, 51L, 51L, 2L))

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

saver <- tf$train$Saver()

sess <- tf$InteractiveSession()

saver$restore(sess, "/srv/scratch/z5016924/y_model/attempt5/model.ckpt")

model.response <- tf$equal(tf$argmax(y, 1L), 0L)


# Import Scans -------------------------------------------------------------
# Rough code just to import scan separately (so reload isn't necessary later)
# Import both soft tissue and flair
# Also import true lacune scan
scan.id <- "4689"

soft <- f.read.nifti.volume(paste0("/srv/scratch/z5016924/MAS_W2/T1softTiss/",scan.id,"_T1softTiss.nii"))
flair <- f.read.nifti.volume(paste0("/srv/scratch/z5016924/MAS_W2/FLAIRinT1space/r",scan.id,"_tp2_flair.nii"))
lacune <- f.read.nifti.volume(paste0("/srv/scratch/z5016924/MAS_W2/lacune_T1space/",scan.id,"_lacuneT1space.nii"))



# Apply model -------------------------------------------------------------
# Function that takes in large scan matrices: soft tissue and flair, and 
# outputs a large matrix of possible lacunes. 1 positive, 0 negative.
# Initialise empty matrix of same dimensions as soft tissue
# Iterate through matrix in 51x51 patches
# For each patch, convert to a single vector of values just as
# with '2_ImportMRI.R'
# Vector will be of length 5202:
# - 51x51 soft tissue
# - 51x51 flair
# Input the actual image data into the model.
# If response is positive, take the x/y/z coordinates and place a 1 in
# the intialised matrix. Else move on to the next point.
# Every 100 points examined, print to screen the progress %
# Use gc() to clear between iterations
# After all iterations, return the matrix of possible lacunes
ApplyModel <- function(soft.tissue, flair){
  # Inputs the two image arrays
  # Outputs an array of same size with 1s for possible lacunes
  output <- array(dim = dim(soft.tissue))
  
  
}

IsPositive <- function(x.coord, y.coord, z.coord, soft.tissue, flair) {
  # Takes in images and coordinates.
  # Extracts 51x51 slices and arranges into vector
  # Feed into model.
  # If positive, return TRUE, else FALSE
  
  sample <- array(dim = c(1, 5202))
  sample[1,1:2601] <- soft.tissue[(x.coord-25):(x.coord+25),(y.coord-25):(y.coord+25), z.coord, 1]
  sample[1,2602:5202] <- flair[(x.coord-25):(x.coord+25),(y.coord-25):(y.coord+25), z.coord, 1]
  
  response <- y$eval(feed_dict = dict(x = sample, keep.prob = 1))
  
  return(response)
}

# [,1] [,2] [,3] [,4]     [,5]
# [1,] 6324  109  161   68 192791.4
# [2,] 4689  100  159   77 178122.2
# [3,] 1183  138  149   76      0.0
# [4,] 1224  100  153   68 183008.9
# [5,] 6324  109  160   69 224562.1
# [6,] 1535  103  149  106 110534.3
#        46  108  149  103


sess$close()
