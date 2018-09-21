
# Data prep ---------------------------------------------------------------

# setwd("/srv/scratch/z5016924/model1/attempt4")
setwd("~/hdrive/Honours/lacune-pvs-cnn/attempt4")

load("train_accuracy.Rda")
load("train_accuracy2.Rda")

train.accuracy2 <- head(train.accuracy2, -1)


# Plots Attempt 4 -------------------------------------------------------------------

which(train.accuracy == max(train.accuracy))[[1]]
# attempt4: First hits 100% at batch 28x5 = 140

which(train.accuracy2 == max(train.accuracy2))[[1]]
max(train.accuracy2)
# attempt4: Maximum validation accuracy at epoch 31, at 0.9899445

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
# attempt 4: Seems to remain stable after about 15 epochs.
# Maximum is reached at epoch 31 with accuracy 0.9899445
abline(h = max(train.accuracy2), col = "blue", lty = 2)
points(31, train.accuracy2[31])
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9899445"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))



# Plots Attempt 5 ---------------------------------------------------------


# setwd("/srv/scratch/z5016924/model1/attempt5")
setwd("~/hdrive/Honours/lacune-pvs-cnn/attempt5")

load("train_accuracy.Rda")
load("train_accuracy2.Rda")

train.accuracy2 <- head(train.accuracy2, -1)


  
which(train.accuracy == max(train.accuracy))[[1]]
# attempt5: First hits 100% at 41x5 = 205

which(train.accuracy2 == max(train.accuracy2))[[1]]
max(train.accuracy2)
# attempt5: Maximum validation accuracy at epoch 22, at 0.9983581

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
abline(h = max(train.accuracy2), col = "blue", lty = 2)
points(22, train.accuracy2[22])
legend('bottomright',
       c("Validation Accuracy","Cubic Spline", "Highest Acc = 0.9983581"),
       lty = c(2,1,2),
       col = c("black","red", "blue"))


