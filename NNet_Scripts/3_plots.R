
# Data prep ---------------------------------------------------------------

# setwd("/srv/scratch/z5016924/model1/attempt5")
setwd("~/hdrive/Honours/lacune-pvs-cnn/y_attempt5")

load("train_accuracy.Rda")
load("train_accuracy2.Rda")

train.accuracy2 <- head(train.accuracy2, -1)


# Plots Attempt 4 -------------------------------------------------------------------

which(train.accuracy == max(train.accuracy))[[1]]
# attempt4: First hits 100% at batch 28x5 = 140

which(train.accuracy2 == max(train.accuracy2))[[1]]
max(train.accuracy2)
# attempt4: Maximum validation accuracy at epoch 18, at 0.9923717

plot((1:length(train.accuracy))*5+1, train.accuracy,
     type = "l", lty = "dashed",
     xlab = "Batch Number",
     ylab = "Training Accuracy",
     main = "Training Accuracy")
# attempt4: Consistently at 100% by batch 300
lines(smooth.spline((1:length(train.accuracy))*5+1, train.accuracy,
                    spar = 0.4),col = "red")
legend("bottomright",
       c("Training Accuracy", "Cubic Spline"),
       lty = c(2, 1),
       col = c("black","red"))


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
points(which.max.acc.4, max.acc.4)
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9923717"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))



# Plots Attempt 5 ---------------------------------------------------------


# setwd("/srv/scratch/z5016924/model1/attempt5")
setwd("~/hdrive/Honours/lacune-pvs-cnn/attempt5")

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
     xlab = "Batch Number",
     ylab = "Training Accuracy",
     main = "Training Accuracy")
# attempt5: More noise - from larger sample size. Consistently at 100% by batch 1000.
lines(smooth.spline((1:length(train.accuracy))*5+1, train.accuracy, spar = 0.4),
      col = "red")
legend("bottomright",
       c("Training Accuracy", "Cubic Spline"),
       lty = c(2, 1),
       col = c("black","red"))


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
points(which.max.acc.5, max.acc.5)
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9983333"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))


