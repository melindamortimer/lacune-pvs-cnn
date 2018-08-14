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
# image(soft[,,140,], col = grey.colors(100))

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

ViewPatch <- function(id, x, y, z, type = "soft") {
  id <- sprintf("%04d", id)
  if (type == "soft") {
    file.name <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  } else if (type == "t1") {
    file.name <- paste(data.dir, "T1/", id, "_tp2_t1.nii", sep = "")
  } else if (type == "flair") {
    file.name <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  } else {
    stop("Type needs to be one of 'soft', 't1' or 'flair'")
  }
  img <- f.read.nifti.volume(file.name)
  subimg <- img[(x - 25):(x + 25), (y - 25):(y + 25), z, 1]
  image(subimg, col = grey.colors(100))
}

ViewSlice <- function(id,z) {
  id <- sprintf("%04d", id)
  file.name <- paste(data.dir, "T1/", id, "_tp2_t1.nii", sep = "")
  img <- f.read.nifti.volume(file.name)
  subimg <- img[,,z, 1]
  image(subimg, col = grey.colors(100))
}



# Import ------------------------------------------------------------------
# image dim: [256,256,190,1]
# sliding screen dim: [206, 206, 190, 1]
# centre: [26:231, 26:231, 1:190, 1]

list.t1 <- list.files("/srv/scratch/z5016924/MAS_W2/T1softTiss/")
list.flair <- list.files("/srv/scratch/z5016924/MAS_W2/FLAIRinT1space/")
list.lacune <- list.files("/srv/scratch/z5016924/MAS_W2/lacune_T1space/")

list.id <- str_extract(list.t1, "^[0-9]{4}")
list.id.lacune <- str_extract(list.lacune, "^[0-9]{4}")

max.rows <- 3000

# large number of samples (max.rows), each is made of:
# - scan id
# - x/y/z
# - 51x51 = 2601 t1
# - 51x51 flair
# - response
# Total of 5207 variables per sample
data.lacunes <- array(NA, dim = c(max.rows, 5207))
# Find lacune samples first
i <- 1
for (id in list.id.lacune) {
# for (id in "0046") {
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

        patch.t1 <- soft[(x - 25):(x + 25), (y - 25):(y + 25), z, 1]
        patch.flair <- flair[(x - 25):(x + 25), (y - 25):(y + 25), z, 1]
        
        data.lacunes[i, 5:2605] <- patch.t1
        data.lacunes[i, 2606:5206] <- patch.flair
        
        data.lacunes[i, 5207] <- 1
        
        i <- i + 1
        if (i > max.rows) break
      }
    }
  }
  
}



# Negatives ---------------------------------------------------------------


# Only scan in spots that can fit a patch.
# ie x 51:dim(soft)[1] - 50
# Try to collect 200 points per scan (1 per z slice?)
# Randomise x/y. If lacune, pass.
# If scan value 0, pass.
max.rows2 <- 100000
data.notlacunes <- array(NA, dim = c(max.rows2, 5207))
i <- 1
for (id in list.id[1]) {
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
  
  print("1")
  # INCORRECT NUMBER OF DIMENSIONS FOR LACUNES BETWEEN 3 AND 4
  # CAREFUL THAT LOOP CURRENTLY WILL RUN FOREVER IF A LAYER DOES NOT HAVE ANY NONZERO READINGS IN T1
  
  for (z in 1:dim(soft)[3]) {
    print("2")
    # Randomise x/y
    x <- round(runif(1,51,(dim(soft)[1] - 50)))
    y <- round(runif(1,51,(dim(soft)[2] - 50)))
    print("3")
    # Keep randomising until the sample is not empty and not a lacune
    while(lacune[x,y,z] == 0 | soft[x,y,z] == 0) {
      print("4")
      x <- round(runif(51:(dim(soft)[1] - 50)))
      y <- round(runif(51:(dim(soft)[2] - 50)))
    }
    print("5")
    print(paste("Lacune at [", x, y, z, "]"))
    data.notlacunes[i, 1] <- as.numeric(id)
    data.notlacunes[i, 2] <- x
    data.notlacunes[i, 3] <- y
    data.notlacunes[i, 4] <- z
    
    patch.t1 <- soft[(x - 25):(x + 25), (y - 25):(y + 25), z, 1]
    patch.flair <- flair[(x - 25):(x + 25), (y - 25):(y + 25), z, 1]
    
    data.notlacunes[i, 5:2605] <- patch.t1
    data.notlacunes[i, 2606:5206] <- patch.flair
    
    data.notlacunes[i, 5207] <- 0
    
    i <- i + 1
    if (i > max.rows2) break
      
  }
  
}
