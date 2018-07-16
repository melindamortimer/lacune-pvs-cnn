library(AnalyzeFMRI)
hdr <- f.read.header("../tenSubjects/0046_t1_MAS_w1.nii")
hdr$dim
# Dimensions of image: 
# 3D
# x: 256 (slice width)
# y: 190 (slice height)
# z: 245 (volume depth)

hdr$datatype
# Datatype 4: Signed short

hdr$pixdim
# voxel width: 1mm
# voxel height: 1mm
# slice thickness: 1mm
# timeslice: 1

A <- f.read.nifti.volume("../tenSubjects/0046_t1_MAS_w1.nii")
dim(A)
image(A[,,150,])

B <- f.read.nifti.slice("../tenSubjects/0046_t1_MAS_w1.nii",
                        slice = 100, tpt = 1)
image(B)