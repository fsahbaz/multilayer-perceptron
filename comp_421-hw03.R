# Defining the safelog function.
safelog <- function(x) {
  return (log(x + 1e-100))
}
# Defining the sigmoid function.
sigmoid <- function(x) {
  return (1 / (1 + exp(-x)))
}
# Defining the softmax function.
softmax <- function(x){
  x <- exp(x)
  return (x / rowSums(x))
}

# Reading the image, label, and weight datum.
data_set <- read.csv(file = "hw03_images.csv", header = FALSE)
labels <- read.csv(file = "hw03_labels.csv", header = FALSE)
W <- as.matrix(read.csv(file = "initial_W.csv", header = FALSE))
v <- as.matrix(read.csv(file = "initial_V.csv", header = FALSE))

# Splitting the read data into train and test data sets.
rows <- nrow(data_set)
cols <- ncol(data_set)

X_train <- as.matrix(data_set[1:(rows/2),1:cols])
X_test <- as.matrix(data_set[(rows/2+1):rows,1:cols])

y_train <- as.numeric(labels[1:(rows/2),1])
y_test <- as.numeric(labels[(rows/2+1):rows,1])

# One-of-K-encoding.
Y_truth_train <- matrix(0, length(y_train), max(y_train))
Y_truth_train[cbind(1:length(y_train), y_train)] <- 1

Y_truth_test <- matrix(0, length(y_test), max(y_test))
Y_truth_test[cbind(1:length(y_test), y_test)] <- 1

# Extracting the number of samples and classes.
N <- nrow(Y_truth_train)
K <- ncol(Y_truth_train)
D <- ncol(X_train)

# Setting the learning parameters.
eta <- 0.0005
epsilon <- 1e-3
H <- 20
max_iteration <- 500

Z <- sigmoid(cbind(1, X_train) %*% W)
Y_pred_train <- softmax(cbind(1, Z) %*% v)
objective_values <- -sum(Y_truth_train * safelog(Y_pred_train))

delta_W <- matrix(0, nrow=D+1, ncol=H)
delta_v <- matrix(0, nrow=H+1, ncol=K)

# Learning W and v using gradient descent and online learning.
iteration <- 1
while (1) {

  for (i in sample(N)) {
    # Computing nodes in the hidden layer.
    Z[i,] <- sigmoid(c(1, X_train[i,]) %*% W)
    
    # Computing nodes in the hidden layer.
    Y_pred_train[i,] <- softmax(c(1, Z[i,]) %*% v)
    
    # Calculating gradient descent for v,
    for (k in 1:K) {
      delta_v[,k] <- eta * (Y_truth_train[i,k] - Y_pred_train[i,k]) * c(1, Z[i,]) 
    }
    
    # Calculating gradient descent for W.
    for (h in 1:H) {
      delta_W[,h] <- eta * sum((Y_truth_train[i,] - Y_pred_train[i,]) * v[h,]) * Z[i,h] * (1 - Z[i,h]) * c(1, X_train[i,])
    }
    
    # Updating W and v
    v <- v + delta_v
    W <- W + delta_W
  }
  
  # Updating predictions according to the new values.
  Z <- sigmoid(cbind(1, X_train) %*% W)
  Y_pred_train <- softmax(cbind(1, Z) %*% v)
  # Calculating error.
  objective_values <- c(objective_values, -sum(Y_truth_train * safelog(Y_pred_train)))
  
  # Terminating condition.
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  iteration <- iteration + 1
}

# Plotting error against iterations.
plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

# Finalizing the predictions and printing the confusion matrix for the training data.
Y_truth_train_fin <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_train[,c]*c}))
Y_pred_train <- apply(Y_pred_train, 1, which.max)
conf_mat_tr <- table(Y_pred_train, Y_truth_train_fin)
print(conf_mat_tr)

# Making predictions from the finalized values of W and v for the test data.
Z <- sigmoid(cbind(1, X_test) %*% W)
Y_pred_test <- softmax(cbind(1, Z) %*% v)

# Finalizing the predictions and printing the confusion matrix for the test data.
Y_truth_test_fin <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_test[,c]*c}))
Y_pred_test <- apply(Y_pred_test, 1, which.max)
conf_mat_te <- table(Y_pred_test, Y_truth_test_fin)
print(conf_mat_te)