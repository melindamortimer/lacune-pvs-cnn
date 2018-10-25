# Evaluation of slice. Outputs probabilities as heat map

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

