library(AnalyzeFMRI)
library(stringr)
library(crayon)

data.dir <- "/srv/scratch/z5016924/MAS_W2/"



# hdr <- f.read.header("../../tenSubjects/0046_t1_MAS_w1.nii")
# hdr$dim
# Dimensions of image: 
# 3D
# x: 256 (slice width)
# y: 190 (slice height)
# z: 245 (volume depth)

# hdr$datatype
# Datatype 4: Signed short

# hdr$pixdim
# voxel width: 1mm
# voxel height: 1mm
# slice thickness: 1mm
# timeslice: 1

# A <- f.read.nifti.volume("../../tenSubjects/0046_t1_MAS_w1.nii")
# dim(A)
# image(A[,,150,], col = grey.colors(100))
# 
# B <- f.read.nifti.slice("../../tenSubjects/0046_t1_MAS_w1.nii",
#                         slice = 100, tpt = 1)
# image(B, col = grey.colors(100))


# t1 <- f.read.nifti.volume("/srv/scratch/z5016924/MAS_W2/T1/0046_tp2_t1.nii")
# flair <- f.read.nifti.volume("/srv/scratch/z5016924/MAS_W2/FLAIRinT1space/r0046_tp2_flair.nii")
# soft <- f.read.nifti.volume("/srv/scratch/z5016924/MAS_W2/T1softTiss/0022_T1softTiss.nii")
# lacune <- f.read.nifti.volume("/srv/scratch/z5016924/MAS_W2/lacune_T1space/0046_lacuneT1space.nii")

# image(t1[,,140,], col = grey.colors(100))
# image(soft[,,138,], col = grey.colors(100))

# image(flair[,,140,], col = grey.colors(100))
# image(lacune[,,140,], col = grey.colors(100))


# Data Structure ----------------------------------------------------------

# Original image data format is as a nifti file:
# - dimensions?
# [256,256,190,1]

# lacune data is currently in an excel sheet. Lacunes will be manually identified within either:
# - excel spreadsheet that lists scan id & x/y/z coords of lacunes. Note all involved pixels
# - a mask of equal dimension that gives value 1 to regions of lacune, 0 otherwise

# First part of the model requires:
# - 51x51 axial patches
# - 0/1 lacune flag
# ID stuff: image id, centre x/y/z

# Second part of the model requires:
# - 32x32x5 blocks
# - 64x64x5 blocks, reshaped to 32x32x5
# - 128x128x5 blocks, reshaped to 32x32x5
# - 0/1 lacune flag
# - Vector of 7 location features:
  # - x/y/z coords
  # - distance to left ventricle
  # - distance to right ventricle
  # - distance to cerebral cortex
  # - distance to midsaggital brain surface

# If lacune data logged via excel, create a data matrix the same dimensions as the scan, with all responses first set to 0. Then go back and iterate through excel sheet, and convert identified pixel responses to 1.

# If lacunes identified by colouring in masks, then the data will be on a separate nifti file of same dimensions as the original image. Import and change to matrix via AnalyzeFMRI.

# Keep each sample together.
# For the first model, each x/y/z is a sample of two variables: the 2D image and the flag.
# For the second model, each x/y/z is a sample of 11 variables: the three 3D images, flag and 7 location features.

# ALTERNATIVELY store for both models by just storing for the second, then retrieving variables for the first by taking the middle slice of the 32x32x5 block.

# View Functions ----------------------------------------------------------
data.dir <- "/srv/scratch/z5016924/MAS_W2/"

ViewPatch <- function(id, x, y, z, type = "soft", point = T, res = 25) {
  id <- sprintf("%04d", id)
  if (type == "soft") {
    file.name <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  } else if (type == "t1") {
    file.name <- paste(data.dir, "T1/", id, "_tp2_t1.nii", sep = "")
  } else if (type == "flair") {
    file.name <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  } else if (type == "lacune") {
    file.name <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  } else {
    stop("Type needs to be one of 'soft', 't1' or 'flair'")
  }
  img <- f.read.nifti.volume(file.name)
  subimg <- img[(x - res):(x + res), y, (z-res):(z+res), 1]
  image(subimg, col = grey.colors(100))
  if (point) points(0.5, 0.5, col = "red")
}

ViewSlice <- function(id,y, type = "soft", point = T, col = grey.colors(100)) {
  id <- sprintf("%04d", id)
  if (type == "soft") {
    file.name <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  } else if (type == "t1") {
    file.name <- paste(data.dir, "T1/", id, "_tp2_t1.nii", sep = "")
  } else if (type == "flair") {
    file.name <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  } else if (type == "lacune") {
    file.name <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  } else {
    stop("Type needs to be one of 'soft', 't1' or 'flair'")
  }
  img <- f.read.nifti.volume(file.name)
  subimg <- img[,y,, 1]
  image(subimg, col = col)
  if (point) points(0.5, 0.5, col = "red")
  
}



# Positives ------------------------------------------------------------------
# image dim: [256,256,190,1]
# sliding screen dim: [206, 206, 190, 1]
# centre: [26:231, 26:231, 1:190, 1]

list.t1 <- list.files("/srv/scratch/z5016924/MAS_W2/T1softTiss/")
list.flair <- list.files("/srv/scratch/z5016924/MAS_W2/FLAIRinT1space/")
list.lacune <- list.files("/srv/scratch/z5016924/MAS_W2/lacune_T1space/")

list.id <- str_extract(list.t1, "^[0-9]{4}")
list.id.lacune <- str_extract(list.lacune, "^[0-9]{4}")


# large number of samples (max.rows), each is made of:
# - scan id
# - x/y/z
# - 51x51 = 2601 t1
# - 51x51 flair
# - response = lacune
# - response = nonlacune
# Total of 5208 variables per sample

max.rows <- 6000
data.lacunes <- array(NA, dim = c(max.rows, 5208))
i <- 1
for (id in list.id.lacune) {
  cat(white$bgBlack(paste("Processing",id,"\n")))

  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)

  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)

  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  lacune <- f.read.nifti.volume(file.lacune)

  for (x in 1:dim(soft)[1]) {
    for (y in 1:dim(soft)[2]) {
      for (z in 1:dim(soft)[3]) {
        if (lacune[x,y,z,1] == 0) next
        print(paste("Lacune at [", x, y, z, "]"))
        data.lacunes[i, 1] <- as.numeric(id)
        data.lacunes[i, 2] <- x
        data.lacunes[i, 3] <- y
        data.lacunes[i, 4] <- z

        patch.t1 <- soft[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]

        data.lacunes[i, 5:2605] <- patch.t1
        data.lacunes[i, 2606:5206] <- patch.flair

        data.lacunes[i, 5207] <- 1
        data.lacunes[i, 5208] <- 0

        i <- i + 1
        if (i > max.rows) break


        # Also save the reverse image
        data.lacunes[i, 1] <- as.numeric(id)
        data.lacunes[i, 2] <- x
        data.lacunes[i, 3] <- y
        data.lacunes[i, 4] <- z

        patch.t1 <- soft[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x + 25):(x - 25), y, (z-25):(z+25), 1]

        data.lacunes[i, 5:2605] <- patch.t1
        data.lacunes[i, 2606:5206] <- patch.flair

        data.lacunes[i, 5207] <- 1
        data.lacunes[i, 5208] <- 0

        i <- i + 1
        if (i > max.rows) break
      }
    }
  }

}

numrows <- max(which(!is.na(data.lacunes[,1])))
data.lacunes <- data.lacunes[1:numrows,]

# Randomise and save
data.lacunes <- data.lacunes[sample(nrow(data.lacunes)),]

save(data.lacunes, file = "/srv/scratch/z5016924/data_lacunes.Rda")



# Negatives ---------------------------------------------------------------


# Only scan in spots that can fit a patch.
# ie x 51:dim(soft)[1] - 50
# Try to collect 200 points per scan (1 per z slice?)
# Randomise x/y. If lacune, pass.
# If scan value 0, pass.
max.rows2 <- 50000
data.nonlacune <- array(NA, dim = c(max.rows2, 5208))
i <- 1
for (id in list.id) {
  cat(white$bgBlack(paste("Processing",id,"\n")))

  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)

  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)

  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  if (file_test("-f", file.lacune)) {
    lacune <- f.read.nifti.volume(file.lacune)
  } else {
    lacune <- array(data = 0, dim = dim(soft))
  }

  # Instead, take every kth pixel of matter? Skip if not matter or if a lacune. Sequence starts randomly between 26 and 76

  for (x in seq(round(runif(1, 26, 76)), dim(soft)[1] - 25, by = 25)) {
    for (y in seq(round(runif(1, 26, 76)), dim(soft)[2] - 25, by = 25)) {
      for (z in seq(round(runif(1, 26, 76)), dim(soft)[3] - 25, by = 25)) {

        # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
        midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
        # Skip if pixel is not in brain matter, or is a lacune
        if (lacune[x,y,z,1] == 1 | midregion == 0) next

        print(paste("Non-lacune at [", x, y, z, "]"))
        data.nonlacune[i, 1] <- as.numeric(id)
        data.nonlacune[i, 2] <- x
        data.nonlacune[i, 3] <- y
        data.nonlacune[i, 4] <- z

        patch.t1 <- soft[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]

        data.nonlacune[i, 5:2605] <- patch.t1
        data.nonlacune[i, 2606:5206] <- patch.flair

        data.nonlacune[i, 5207] <- 0
        data.nonlacune[i, 5208] <- 1

        i <- i + 1
        if (i > max.rows2) {
          stop(paste("Reached max number of rows", max.rows2))
        }
      }
    }
  }
}
# 
# # Remove empty rows on end
# 
numrows <- min(which(is.na(data.nonlacune[,1]))) - 1
data.nonlacune <- data.nonlacune[1:numrows,]
# 
# # Randomise and save
# 
data.nonlacune <- data.nonlacune[sample(nrow(data.nonlacune)),]
save(data.nonlacune, file = "/srv/scratch/z5016924/data_nonlacune.Rda")





# Sampling 1 ----------------------------------------------------------------
# Training and testing only.
# load("/srv/scratch/z5016924/data_lacunes.Rda")
# load("/srv/scratch/z5016924/data_nonlacune.Rda")

# dim(data.lacunes)
# [1] 3846 5208

# dim(data.nonlacune)
# [1] 40008  5208

# Paper has 320K patches in total. 2/3 of these are negatives
# In our data, only 7% are positives, with around 52000 samples in total.
# Split training and testing to 70:30

# 51822 rows total
# training: 2692 pos, 33583 neg. 36275 total
# testing: 1154 pos, 14393 neg. 15544 total

# training <- rbind(data.lacunes[1:2692,], data.nonlacune[1:33583,])
# training <- training[sample(36275),]
# 
# testing <- rbind(data.lacunes[2693:3846,], data.nonlacune[33584:47976,])
# testing <- testing[sample(15544),]

# dim(training)
# [1] 36275  5208
# dim(testing)
# [1] 15544  5208

# 7.4% positives
# save(training, file = "/srv/scratch/z5016924/training.Rda") 
# save(testing, file = "/srv/scratch/z5016924/testing.Rda")




# Sampling 2 --------------------------------------------------------------
# 1/3 positives: training, validation and testing sets
# load("/srv/scratch/z5016924/data_lacunes.Rda")
# load("/srv/scratch/z5016924/data_nonlacune.Rda")

# To get the 1/3 positives seen in the paper
# Split data training:validation:testing = 50:25:25
data.nonlacune2 <- data.nonlacune[sample(nrow(data.nonlacune),2*nrow(data.lacunes)),]
data2 <- rbind(data.lacunes, data.nonlacune2)
n <- nrow(data2)
data2 <- data2[sample(n),]

training <- data2[1:floor(n/2),]
validation <- data2[(floor(n/2)+1):floor(0.75*n),]
testing <- data2[(floor(0.75*n)+1):n,]

save(training, file = "/srv/scratch/z5016924/training2.Rda")
save(validation, file = "/srv/scratch/z5016924/validation2.Rda")
save(testing, file = "/srv/scratch/z5016924/testing2.Rda")


dim(training)
# [1] 5769 5208
dim(validation)
#[1] 2884 5208
dim(testing)
#[1] 2885 5208


# Sampling 3 --------------------------------------------------------------

# Use all samples, keeping the positive percentage at a low percentage.
# Training, validation and testing sets
# Not too low, in case the model starts outputting non lacune too often, just to inflate accuracy.
dim(data.lacunes) # [1] 3846 5208
dim(data.nonlacune) # [1] 39983  5208
# 43829 samples total
# Lacunes now make 8.78% of data samples.

load("/srv/scratch/z5016924/data_lacunes.Rda")
load("/srv/scratch/z5016924/data_nonlacune.Rda")

# Split into training/validation/testing = 50:25:25
nl <- nrow(data.lacunes)
nn <- nrow(data.nonlacune)

training <- rbind(data.lacunes[1:floor(0.5*nl),], data.nonlacune[1:floor(0.5*nn),])
training <- training[sample(nrow(training)),]

validation <- rbind(data.lacunes[(floor(0.5*nl)+1):floor(0.75*nl),], data.nonlacune[(floor(0.5*nn)+1):floor(0.75*nn),])
validation <- validation[sample(nrow(validation)),]

testing <- rbind(data.lacunes[(floor(0.75*nl)+1):nl,], data.nonlacune[(floor(0.75*nn)+1):nn,])
testing <- testing[sample(nrow(testing)),]

dim(training)
# [1] 21914  5208

dim(validation)
# [1] 10957  5208

dim(testing)
# [1] 10958  5208

save(training, file = "/srv/scratch/z5016924/training3.Rda")
save(validation, file = "/srv/scratch/z5016924/validation3.Rda")
save(testing, file = "/srv/scratch/z5016924/testing3.Rda")
