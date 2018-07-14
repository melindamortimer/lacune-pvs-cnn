library(tensorflow)

# Create 100 phony x, y data points, y = x * 0.1 + 0.3
x_data <- runif(100, min=0, max=1)
y_data <- x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# (We know that W should be 0.1 and b 0.3, but TensorFlow will
# figure that out for us.)
W <- tf$Variable(tf$random_uniform(shape(1L), -1.0, 1.0))
b <- tf$Variable(tf$zeros(shape(1L)))
y <- W * x_data + b

# Minimize the mean squared errors.
loss <- tf$reduce_mean((y - y_data) ^ 2)
optimizer <- tf$train$GradientDescentOptimizer(0.5)
train <- optimizer$minimize(loss)
best.loss <- tf$Variable(Inf, shape(1L))

# saver <- tf$train$Saver(list(W, b, best.loss))

# Launch the graph and initialize the variables.
sess = tf$InteractiveSession()
sess$run(tf$global_variables_initializer())

for (step in 1:10001) {
  sess$run(train)
  if (step %% 20 == 0) {
    cat(step, "-", sess$run(W), sess$run(b), "\n")
    # if (sess$run(loss) < sess$run(best.loss)) {
    #   sess$run(tf$assign(best.loss, loss))
    #   saver$save(sess, './my-model/model', global_step = step)
    # }
  }
}

# sess$run(W)
# 
# saver$restore(sess, tf$train$latest_checkpoint('./my-model/'))

# sess$run(W)

sess$close()

