library(AnalyzeFMRI)
library(stringr)
library(crayon)

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



# Gather files and ids -------------------------------------------------------

list.t1 <- list.files("/srv/scratch/z5016924/MAS_W2/T1softTiss/")
list.flair <- list.files("/srv/scratch/z5016924/MAS_W2/FLAIRinT1space/")
list.lacune <- list.files("/srv/scratch/z5016924/MAS_W2/lacune_T1space/")

list.id <- str_extract(list.t1, "^[0-9]{4}")
# id of lacune samples
list.id.lacune <- str_extract(list.lacune, "^[0-9]{4}")

# id of nonlacune samples
list.id.nonlacune <- setdiff(list.id, list.id.lacune)

# Split id of lacune and nonlacune samples into three sets
list.id.lacune.sets <- list(list.id.lacune[1:18], list.id.lacune[19:27], list.id.lacune[28:35])
list.id.nonlacune.sets <- list(list.id.nonlacune[1:188], list.id.nonlacune[189:282],
                               list.id.nonlacune[283:376])


# Train - Positive --------------------------------------------------------
# Train/valid/test index
tvt <- 1
# For each set, generate positive samples + flipped samples
max.rows <- 6000
data.train.lacunes <- array(NA, dim = c(max.rows, 5208))
i <- 1
for (id in list.id.lacune.sets[[tvt]]) {
  cat(white$bgBlack(paste("Processing",id,"\n")))
  
  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)
  
  file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
  t1 <- f.read.nifti.volume(file.t1)
  
  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)
  
  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  lacune <- f.read.nifti.volume(file.lacune)
  
  for (x in 1:dim(soft)[1]) {
    for (y in 1:dim(soft)[2]) {
      for (z in 1:dim(soft)[3]) {
        if (lacune[x,y,z,1] == 0) next
        print(paste("Lacune at [", x, y, z, "]"))
        data.train.lacunes[i, 1] <- as.numeric(id)
        data.train.lacunes[i, 2] <- x
        data.train.lacunes[i, 3] <- y
        data.train.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.train.lacunes[i, 5:2605] <- patch.t1
        data.train.lacunes[i, 2606:5206] <- patch.flair
        
        data.train.lacunes[i, 5207] <- 1
        data.train.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
        
        
        # Also save the reverse image
        data.train.lacunes[i, 1] <- as.numeric(id)
        data.train.lacunes[i, 2] <- x
        data.train.lacunes[i, 3] <- y
        data.train.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        
        data.train.lacunes[i, 5:2605] <- patch.t1
        data.train.lacunes[i, 2606:5206] <- patch.flair
        
        data.train.lacunes[i, 5207] <- 1
        data.train.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
      }
    }
  }
  
}
# Remove unfilled rows in array.
numrows <- max(which(!is.na(data.train.lacunes[,1])))
data.train.lacunes <- data.train.lacunes[1:numrows,]
# dim [1] 2538 5208

# Train - Negative ----------------------------------------------------------

max.rows2 <- 50000
data.train.nonlacunes <- array(NA, dim = c(max.rows2, 5208))
i <- 1
for (id in c(list.id.lacune.sets[[tvt]], list.id.lacune.sets[[tvt]])) {
  cat(white$bgBlack(paste("Processing",id,"\n")))
  
  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)
  
  file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
  t1 <- f.read.nifti.volume(file.t1)
  
  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)
  
  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  if (file_test("-f", file.lacune)) {
    lacune <- f.read.nifti.volume(file.lacune)
  } else {
    lacune <- array(data = 0, dim = dim(soft))
  }
  
  # Instead, take every kth pixel of matter? Skip if not matter or if a lacune. Sequence starts randomly between 26 and 76
  
  for (x in seq(round(runif(1, 26, 56)), dim(soft)[1] - 26, by = 15)) {
    for (y in seq(round(runif(1, 26, 56)), dim(soft)[2] - 26, by = 15)) {
      for (z in seq(round(runif(1, 26, 56)), dim(soft)[3] - 26, by = 15)) {
        
        # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
        midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
        # Skip if pixel is not in brain matter, or is a lacune
        if (lacune[x,y,z,1] == 1 | midregion == 0) next
        
        print(paste("Non-lacune at [", x, y, z, "]"))
        data.train.nonlacunes[i, 1] <- as.numeric(id)
        data.train.nonlacunes[i, 2] <- x
        data.train.nonlacunes[i, 3] <- y
        data.train.nonlacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.train.nonlacunes[i, 5:2605] <- patch.t1
        data.train.nonlacunes[i, 2606:5206] <- patch.flair
        
        data.train.nonlacunes[i, 5207] <- 0
        data.train.nonlacunes[i, 5208] <- 1
        
        i <- i + 1
        if (i > max.rows2) {
          stop(paste("Reached max number of rows", max.rows2))
        }
      }
    }
  }
}

# Remove empty rows on end
numrows <- min(which(is.na(data.train.nonlacunes[,1]))) - 1
data.train.nonlacunes <- data.train.nonlacunes[1:numrows,]
# dim [1] 17363  5208



# Combine Training --------------------------------------------------------

# Train Set 1: All samples
training <- rbind(data.train.lacunes, data.train.nonlacunes)
# Randomise and save
training <- training[sample(nrow(training)),]

save(data.train.lacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/train_lacunes.Rda")
save(data.train.nonlacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/train_nonlacunes.Rda")
save(training, file = "/srv/scratch/z5016924/Data sets/attempt2/training.Rda")




# Valid - Positive --------------------------------------------------------
# Train/valid/test index
tvt <- 2
# For each set, generate positive samples + flipped samples
max.rows <- 6000
data.valid.lacunes <- array(NA, dim = c(max.rows, 5208))
i <- 1
for (id in list.id.lacune.sets[[tvt]]) {
  cat(white$bgBlack(paste("Processing",id,"\n")))
  
  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)
  
  file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
  t1 <- f.read.nifti.volume(file.t1)
  
  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)
  
  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  lacune <- f.read.nifti.volume(file.lacune)
  
  for (x in 1:dim(soft)[1]) {
    for (y in 1:dim(soft)[2]) {
      for (z in 1:dim(soft)[3]) {
        if (lacune[x,y,z,1] == 0) next
        print(paste("Lacune at [", x, y, z, "]"))
        data.valid.lacunes[i, 1] <- as.numeric(id)
        data.valid.lacunes[i, 2] <- x
        data.valid.lacunes[i, 3] <- y
        data.valid.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.valid.lacunes[i, 5:2605] <- patch.t1
        data.valid.lacunes[i, 2606:5206] <- patch.flair
        
        data.valid.lacunes[i, 5207] <- 1
        data.valid.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
        
        
        # Also save the reverse image
        data.valid.lacunes[i, 1] <- as.numeric(id)
        data.valid.lacunes[i, 2] <- x
        data.valid.lacunes[i, 3] <- y
        data.valid.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        
        data.valid.lacunes[i, 5:2605] <- patch.t1
        data.valid.lacunes[i, 2606:5206] <- patch.flair
        
        data.valid.lacunes[i, 5207] <- 1
        data.valid.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
      }
    }
  }
  
}
# Remove unfilled rows in array.
numrows <- max(which(!is.na(data.valid.lacunes[,1])))
data.valid.lacunes <- data.valid.lacunes[1:numrows,]
# dim [1]  934 5208


# Valid - Negative ----------------------------------------------------------

max.rows2 <- 25000
data.valid.nonlacunes <- array(NA, dim = c(max.rows2, 5208))
i <- 1
for (id in c(list.id.lacune.sets[[tvt]], list.id.lacune.sets[[tvt]])) {
  cat(white$bgBlack(paste("Processing",id,"\n")))
  
  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)
  
  file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
  t1 <- f.read.nifti.volume(file.t1)
  
  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)
  
  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  if (file_test("-f", file.lacune)) {
    lacune <- f.read.nifti.volume(file.lacune)
  } else {
    lacune <- array(data = 0, dim = dim(soft))
  }
  
  # Instead, take every kth pixel of matter? Skip if not matter or if a lacune. Sequence starts randomly between 26 and 76
  
  for (x in seq(round(runif(1, 26, 56)), dim(soft)[1] - 26, by = 15)) {
    for (y in seq(round(runif(1, 26, 56)), dim(soft)[2] - 26, by = 15)) {
      for (z in seq(round(runif(1, 26, 56)), dim(soft)[3] - 26, by = 15)) {
        
        # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
        midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
        # Skip if pixel is not in brain matter, or is a lacune
        if (lacune[x,y,z,1] == 1 | midregion == 0) next
        
        print(paste("Non-lacune at [", x, y, z, "]"))
        data.valid.nonlacunes[i, 1] <- as.numeric(id)
        data.valid.nonlacunes[i, 2] <- x
        data.valid.nonlacunes[i, 3] <- y
        data.valid.nonlacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.valid.nonlacunes[i, 5:2605] <- patch.t1
        data.valid.nonlacunes[i, 2606:5206] <- patch.flair
        
        data.valid.nonlacunes[i, 5207] <- 0
        data.valid.nonlacunes[i, 5208] <- 1
        
        i <- i + 1
        if (i > max.rows2) {
          stop(paste("Reached max number of rows", max.rows2))
        }
      }
    }
  }
}

# Remove empty rows on end
numrows <- min(which(is.na(data.valid.nonlacunes[,1]))) - 1
data.valid.nonlacunes <- data.valid.nonlacunes[1:numrows,]
# dim [1] 8153 5208




# Combine Valid --------------------------------------------------------

# Train Set 1: All samples
validation <- rbind(data.valid.lacunes, data.valid.nonlacunes)
# Randomise and save
validation <- validation[sample(nrow(validation)),]

save(data.valid.lacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/valid_lacunes.Rda")
save(data.valid.nonlacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/valid_nonlacunes.Rda")
save(validation, file = "/srv/scratch/z5016924/Data sets/attempt2/validation.Rda")


# Test - Positive --------------------------------------------------------
# Train/valid/test index
tvt <- 3
# For each set, generate positive samples + flipped samples
max.rows <- 6000
data.test.lacunes <- array(NA, dim = c(max.rows, 5208))
i <- 1
for (id in list.id.lacune.sets[[tvt]]) {
  cat(white$bgBlack(paste("Processing",id,"\n")))
  
  file.soft <- paste(data.dir, "T1softTiss/", id, "_T1softTiss.nii", sep = "")
  soft <- f.read.nifti.volume(file.soft)
  
  file.t1 <- paste0(data.dir, "T1/", id, "_tp2_t1.nii")
  t1 <- f.read.nifti.volume(file.t1)
  
  file.flair <- paste(data.dir, "FLAIRinT1space/r", id, "_tp2_flair.nii", sep = "")
  flair <- f.read.nifti.volume(file.flair)
  
  file.lacune <- paste(data.dir, "lacune_T1space/", id, "_lacuneT1space.nii", sep = "")
  lacune <- f.read.nifti.volume(file.lacune)
  
  for (x in 1:dim(soft)[1]) {
    for (y in 1:dim(soft)[2]) {
      for (z in 1:dim(soft)[3]) {
        if (lacune[x,y,z,1] == 0) next
        print(paste("Lacune at [", x, y, z, "]"))
        data.test.lacunes[i, 1] <- as.numeric(id)
        data.test.lacunes[i, 2] <- x
        data.test.lacunes[i, 3] <- y
        data.test.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.test.lacunes[i, 5:2605] <- patch.t1
        data.test.lacunes[i, 2606:5206] <- patch.flair
        
        data.test.lacunes[i, 5207] <- 1
        data.test.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
        
        
        # Also save the reverse image
        data.test.lacunes[i, 1] <- as.numeric(id)
        data.test.lacunes[i, 2] <- x
        data.test.lacunes[i, 3] <- y
        data.test.lacunes[i, 4] <- z
        
        patch.t1 <- t1[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x + 25):(x - 25), y, (z-25):(z+25), 1]
        
        data.test.lacunes[i, 5:2605] <- patch.t1
        data.test.lacunes[i, 2606:5206] <- patch.flair
        
        data.test.lacunes[i, 5207] <- 1
        data.test.lacunes[i, 5208] <- 0
        
        i <- i + 1
        if (i > max.rows) break
      }
    }
  }
  
}
# Remove unfilled rows in array.
numrows <- max(which(!is.na(data.test.lacunes[,1])))
data.test.lacunes <- data.test.lacunes[1:numrows,]
# dim [1]  374 5208


# Test - Negative ----------------------------------------------------------

max.rows2 <- 25000
data.test.nonlacunes <- array(NA, dim = c(max.rows2, 5208))
i <- 1
for (id in c(list.id.lacune.sets[[tvt]], list.id.lacune.sets[[tvt]])) {
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
  
  for (x in seq(round(runif(1, 26, 56)), dim(soft)[1] - 26, by = 15)) {
    for (y in seq(round(runif(1, 26, 56)), dim(soft)[2] - 26, by = 15)) {
      for (z in seq(round(runif(1, 26, 56)), dim(soft)[3] - 26, by = 15)) {
        
        # Isolate a 5x5 square in the middle of the sample. If the whole square is 0, skip
        midregion <- sum(soft[(x-4):(x+4), (y-4):(y+4), (z-4):(z+4),1])
        # Skip if pixel is not in brain matter, or is a lacune
        if (lacune[x,y,z,1] == 1 | midregion == 0) next
        
        print(paste("Non-lacune at [", x, y, z, "]"))
        data.test.nonlacunes[i, 1] <- as.numeric(id)
        data.test.nonlacunes[i, 2] <- x
        data.test.nonlacunes[i, 3] <- y
        data.test.nonlacunes[i, 4] <- z
        
        patch.t1 <- t1[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        patch.flair <- flair[(x - 25):(x + 25), y, (z-25):(z+25), 1]
        
        data.test.nonlacunes[i, 5:2605] <- patch.t1
        data.test.nonlacunes[i, 2606:5206] <- patch.flair
        
        data.test.nonlacunes[i, 5207] <- 0
        data.test.nonlacunes[i, 5208] <- 1
        
        i <- i + 1
        if (i > max.rows2) {
          stop(paste("Reached max number of rows", max.rows2))
        }
      }
    }
  }
}

# Remove empty rows on end
numrows <- min(which(is.na(data.test.nonlacunes[,1]))) - 1
data.test.nonlacunes <- data.test.nonlacunes[1:numrows,]
# dim [1] 7903 5208


# Combine Test --------------------------------------------------------

# Train Set 1: All samples
testing <- rbind(data.test.lacunes, data.test.nonlacunes)
# Randomise and save
testing <- testing[sample(nrow(testing)),]

save(data.test.lacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/test_lacunes.Rda")
save(data.test.nonlacunes, file = "/srv/scratch/z5016924/Data sets/attempt2/test_nonlacunes.Rda")
save(testing, file = "/srv/scratch/z5016924/Data sets/attempt2/testing.Rda")
