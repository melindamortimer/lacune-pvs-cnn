# Data prep ---------------------------------------------------------------

# setwd("/srv/scratch/z5016924/model1/attempt5")
# setwd("~/hdrive/Honours/lacune-pvs-cnn/y_attempt5")
# 
# load("train_accuracy.Rda")
# load("train_accuracy2.Rda")
# 
# train.accuracy2 <- head(train.accuracy2, -1)


# Plots Attempt 4 -------------------------------------------------------------------

setwd("~/hdrive/Honours/lacune-pvs-cnn/y_attempt4")

load("train_accuracy.Rda")
load("train_accuracy2.Rda")

train.accuracy2 <- head(train.accuracy2, -1)

which(train.accuracy == max(train.accuracy))[[1]]
# attempt4: First hits 100% at batch 28x5 = 140

which(train.accuracy2 == max(train.accuracy2))[[1]]
max(train.accuracy2)
# attempt4: Maximum validation accuracy at epoch 18, at 0.9923717

plot((1:length(train.accuracy))*5+1, train.accuracy,
     type = "l", lty = "dashed",
     xlab = "Batch Number (log)",
     ylab = "Training Accuracy",
     main = "Training Accuracy",
     log = "x",
     col = "gray50")
# attempt4: Consistently at 100% by batch 300
lines(smooth.spline((1:length(train.accuracy))*5+1, train.accuracy,
                    spar = 0.4),col = "red", lwd= 2)
legend("bottomright",
       c("Training Accuracy", "Cubic Spline"),
       lty = c(2, 1),
       lwd = c(1,2),
       col = c("gray50","red"))


plot(train.accuracy2,
     type = "l", lty = "dashed",
     xlab = "Epoch",
     ylab = "Validation Accuracy",
     main = "Validation Accuracy",
     ylim = c(0.93, .995))
lines(smooth.spline(train.accuracy2, spar = 0.5), col = "red")
(max.acc.4 <- max(train.accuracy2))
(which.max.acc.4 <- which(train.accuracy2 == max(train.accuracy2))[1])
# attempt 4: Seems to remain stable after about 15 epochs.
# Maximum is reached at epoch 18 with accuracy 0.9923717
abline(h = max.acc.4, col = "blue", lty = 2)
points(which.max.acc.4, max.acc.4, col = "green3",pch = 19)
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9924"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))



# Plots Attempt 5 ---------------------------------------------------------


# setwd("/srv/scratch/z5016924/model1/attempt5")
setwd("~/hdrive/Honours/lacune-pvs-cnn/y_attempt5")

load("train_accuracy.Rda")
load("train_accuracy2.Rda")

train.accuracy2 <- head(train.accuracy2, -1)


  
which(train.accuracy == max(train.accuracy))[[1]]
# attempt5: First hits 100% at 43x5 = 215

which(train.accuracy2 == max(train.accuracy2))[[1]]
max(train.accuracy2)
# attempt5: Maximum validation accuracy at epoch 28, at 0.9983333

plot((1:length(train.accuracy))*5+1, train.accuracy,
     type = "l", lty = "dashed",
     xlab = "Batch Number (log)",
     ylab = "Training Accuracy",
     main = "Training Accuracy",
     log = "x",
     col = "gray50")
# attempt5: More noise - from larger sample size. Consistently at 100% by batch 1000.
lines(smooth.spline((1:length(train.accuracy))*5+1, train.accuracy, spar = 0.4),
      col = "red", lwd = 2)
legend("bottomright",
       c("Training Accuracy", "Cubic Spline"),
       lty = c(2, 1),
       lwd = c(1,2),
       col = c("gray50","red"))


plot(train.accuracy2,
     type = "l", lty = "dashed",
     xlab = "Epoch",
     ylab = "Validation Accuracy",
     main = "Validation Accuracy",
     ylim = c(0.965, 1))
lines(smooth.spline(train.accuracy2, spar = 0.5), col = "red")
# Attempt5: small rise from epoch 10 to 20. Some instability from 20 to 30. Then stable again. Staying around 
(max.acc.5 = max(train.accuracy2))
(which.max.acc.5 = which(train.accuracy2 == max.acc.5)[1] )
abline(h = max.acc.5, col = "blue", lty = 2)
points(which.max.acc.5, max.acc.5, col = "green3", pch = 19)
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9983"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))



# T1 vs FLAIR vs lacune  -------------------------------------------------------------

#dev.off()

par(mfrow = c(2,2), mar = c(2.1,2.1,1.5,0.5))
ViewSlice(7921, 144, point = F, type = "soft")
title(main = "T1-Weighted")
ViewSlice(7921, 144, point = F, type = "flair")
title(main = "FLAIR")
ViewSlice(7921, 144, point = F, type = "lacune", col = c("grey","red"))
title(main = "Lacune")




# Positive Samples: Soft + FLAIR --------------------------------------------

par(mfrow = c(2,2), mar = c(2.1,2.1,1.5,0.5))
ViewPatch(1224, 98, 147, 77, point = T, type = "soft")
title(main = "Positive: T1")
ViewPatch(1224, 98, 147, 77, point = T, type = "flair")
title(main = "Positive: FLAIR")
ViewPatch(8609, 140, 183, 87, point = T, type = "soft")
title(main = "Positive: T1")
ViewPatch(8609, 140, 183, 87, point = T, type = "flair")
title(main = "Positive: FLAIR")


# Negative Samples: Soft + FLAIR --------------------------------------------

par(mfrow = c(2,2), mar = c(2.1,2.1,1.5,0.5))
ViewPatch(8651,126,144,147, point = T, type = "soft")
title(main = "Negative: T1")
ViewPatch(8651,126,144,147, point = T, type = "flair")
title(main = "Negative: FLAIR")
ViewPatch(4532,135,188,119, point = T, type = "soft")
title(main = "Negative: T1")
ViewPatch(4532,135,188,119, point = T, type = "flair")
title(main = "Negative: FLAIR")




# Positive sample gif -----------------------------------------------------

# for (i in 140:160) {
#   print(i)
#   png(file = paste0("../../lacune_gif/img",i,".png"))
#   ViewPatch(1224, 98, i, 77, point = F, type = "t1", res = 40)
#   dev.off()
# }
par(mfrow = c(1,2))
ViewPatch(1224, 98, 147, 77, point = F, type = "t1", res = 40)
title(main = "T1-weighted")
ViewPatch(1224, 98, 147, 77, point = F, type = "flair", res = 40)
title(main = "FLAIR")
  

# Detected ----------------------------------------------------------------

# IMAGE DIMENSIONS 900 X 500
# True Positive
par(mfrow = c(1,2))
ViewPatch(2777, 153, 177, 85, type = "soft")
title(main = "True-Positive T1")
ViewPatch(2777, 153, 177, 85, type = "flair")
title(main = "True-Positive FLAIR")

# False Positive
ViewPatch(8743, 141, 199, 108, type = "soft")
title(main = "False-Positive T1")
ViewPatch(8743, 141, 199, 108, type = "flair")
title(main = "False-Positive FLAIR")

ViewPatch(4612,146,171,93, type = "soft")
title(main = "False-Positive T1")
ViewPatch(4612,146,171,93, type = "flair")
title(main = "False-Positive FLAIR")

# False Negative
ViewPatch(8840,144,151,109, type = "soft")
title(main = "False-Negative T1")
ViewPatch(8840,144,151,109, type = "flair")
title(main = "False-Negative FLAIR")

