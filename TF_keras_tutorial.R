####### Installing tensorflow for R #########

install.packages("tensorflow")

library(tensorflow)
install_tensorflow()

## confirm installatioin 
tf$constant("Hello Tensorflow")

## result: tf.Tensor(b'Hellow Tensorflow', shape=(), dtype=string)


###### Installing Keras ##########

install.packages("keras")

library(keras)

##### Testing with MNIST data set ########

mnist <- dataset_mnist()


## normalizing data set
mnist$train$x <- mnist$train$x/255
mnist$test$x <- mnist$test$x/255


## define model using sequential API
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(28, 28)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dropout(0.2) %>% 
  layer_dense(10, activation = "softmax")

## Summary of model
summary(model)

## Model: "sequential"  -- means  it has a sequence of layers
## ___________________________________________________________________________
## Layer (type)                     Output Shape                  Param #     
## ===========================================================================
## flatten (Flatten)                (None, 784)                   0           
## ___________________________________________________________________________
## dense (Dense)                    (None, 128)                   100480        --parms = (28 * 28 + 1) * 128
## ___________________________________________________________________________
## dropout (Dropout)                (None, 128)                   0             -- regularization method :
## ___________________________________________________________________________     instructs severl units to drop out
## dense_1 (Dense)                  (None, 10)                    1290        
## ===========================================================================
## Total params: 101,770
## Trainable params: 101,770
## Non-trainable params: 0
## ___________________________________________________________________________


## building the model by compiling.
model %>% 
  compile(
    loss = "sparse_categorical_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

## Fit model
model %>% 
  fit(
    x = mnist$train$x, y = mnist$train$y,
    epochs = 5,
    validation_split = 0.3,
    verbose = 2
  )
## verbose = 2 , instruction to print metrics

## Train on 42000 samples, validate on 18000 samples
## Epoch 1/5
## 42000/42000 - 3s - loss: 0.3442 - accuracy: 0.9008 - val_loss: 0.1780 - val_accuracy: 0.9484
## Epoch 2/5
## 42000/42000 - 3s - loss: 0.1682 - accuracy: 0.9498 - val_loss: 0.1356 - val_accuracy: 0.9599
## Epoch 3/5
## 42000/42000 - 3s - loss: 0.1242 - accuracy: 0.9626 - val_loss: 0.1233 - val_accuracy: 0.9622
## Epoch 4/5
## 42000/42000 - 3s - loss: 0.0999 - accuracy: 0.9697 - val_loss: 0.1072 - val_accuracy: 0.9685
## Epoch 5/5
## 42000/42000 - 3s - loss: 0.0834 - accuracy: 0.9739 - val_loss: 0.0966 - val_accuracy: 0.9731

#### validation loss continues to go down, so we know we havent overfit yet

## make predictions with model and predict function
predictions <- predict(model, mnist$test$x)

# shows top two predictions of fit model object, predictions
head(predictions, 2)

##              [,1]         [,2]         [,3]         [,4]         [,5]
## [1,] 1.079081e-07 1.105458e-08 4.597065e-05 2.821549e-04 5.768893e-11
## [2,] 2.735454e-06 6.786310e-04 9.992226e-01 8.388522e-05 3.788405e-13
##              [,6]         [,7]         [,8]         [,9]        [,10]
## [1,] 5.044960e-07 3.673492e-14 9.996552e-01 4.329958e-07 1.558235e-05
## [2,] 4.735405e-08 1.990466e-07 3.531684e-11 1.182519e-05 3.717427e-13

### the probabilty for the first two test samples for each of the 10 different classes

## check the labels for the first two
head(mnist$test$y, 2)
## 7 2

## model performance on a different dataset using the evaluate function
model %>% 
  evaluate(mnist$test$x, mnist$test$y, verbose = 0)

## $loss
## [1] 0.0833252
## 
## $accuracy
## [1] 0.9741

############ Saving model : not sure if this works ###########

## To save Keras models for later prediction, 
## you need to use specialized functions, like save_model_tf
save_model_tf(object = model, filepath = "model")


## You can then reload the model and make predictions with:
reloaded_model <- load_model_tf("model")
all.equal(predict(model, mnist$test$x), predict(reloaded_model, mnist$test$x))

## [1] TRUE



######## Converting Spam data set for Learning via Keras in R ###########

if(!file.exists("spam.data")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",
    "spam.data")
}

spam.dt <- data.table::fread("spam.data")

## must convert into an array to work in R
## array documentation
?array

label.col <- ncol(spam.dt)
y <- array(spam.dt[[label.col]], nrow(spam.dt))

set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

X.sc <- scale(spam.dt[, -label.col, with=FALSE])
## -label.col : all the other cols except from label call
## with=FALSE : dont want col name "-label.col" , look for var name "label.col"
## X.sc is numberic matrix

X.train.mat <- X.sc[is.train,]
X.test.mat <- X.sc[is.test,]
X.train.arr <- array(X.train.mat , dim(X.train.mat))
X.test.arr <- array(X.test.mat , dim(X.test.mat))

y.train <- y[is.train]
y.test <- y[is.test]



####### Basic training with TF/keras #########

model <- keras_model_sequential() %>%
  layer_flatten( input_shape = ncol(X.train.mat))  %>%                        # input layer
  layer_dense(units = 100 , activation = "sigmoid" , use_bias = FALSE ) %>%   # hidden layer
  layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )           # ouput layer

model %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

model %>% 
  fit(
    x = X.train.mat, y = y.train,
    epochs = 60,
    validation_split = 0.3,
    verbose = 2
  )

result <- model %>%
  fit(
    x = X.train.mat , y= y.train,
    epochs = 60,
    Validation_split = 0.3,
    verbose = 2
  )

plot(result)

######### mean loss validation ###########

n.splits <- 10
split.metrics.list <-list()
for( split.i in 1:n.splits){
  model <- keras_model_sequential() %>%
    layer_flatten( input_shape = ncol(X.train.mat))  %>%                        # input layer
    layer_dense(units = 100 , activation = "sigmoid" , use_bias = FALSE ) %>%   # hidden layer
    layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )           # ouput layer
  
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  result <- model %>% 
    fit(
      x = X.train.mat, y = y.train,
      epochs = 60,
      validation_split = 0.3,
      verbose = 2
    )

  print(plot(result))
  metrics.wide <- do.call(data.table::data.table, result$metrics)
  metrics.wide[, epoch := 1:.N]
  split.metrics.list[[split.i]] <- data.table::data.table(
    split.i, metrics.wide )
}

split.metrics <- do.call(rbind, split.metrics.list)

split.means <- split.metrics[, .(
  mean.val.loss=mean(val_loss),
  sd.val.loss = sd(val_loss)
), by=epoch]

min.dt <- split.means[which.min(mean.val.loss)]
min.dt[, point := "min"]

library(ggplot2)

ggplot()+
  geom_ribbon(aes(
    x=epoch, ymin=mean.val.loss-sd.val.loss , ymax=mean.val.loss+sd.val.loss),
    alpha=0.5,
    data=split.means) +
  geom_point(aes(
    x=epoch, y=mean.val.loss),
    data=split.means) +
  geom_point(aes(
    x=epoch, y=mean.val.loss, color=point),
    data=min.dt)


######## Test Set Accuracy ##########

model <- keras_model_sequential() %>%
  layer_flatten( input_shape = ncol(X.train.mat))  %>%                        # input layer
  layer_dense(units = 100 , activation = "sigmoid" , use_bias = FALSE ) %>%   # hidden layer
  layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )           # ouput layer

model %>% 
  compile(
    loss = "binary_crossentropy",
    optimizer = "sgd",
    metrics = "accuracy"
  )

result <- model %>% 
  fit(
    x = X.train.mat, y = y.train,
    epochs = min.dt$epoch,
    validation_split = 0,
    verbose = 2
  )

plot(result)


model %>%
  evaluate( X.test.mat , y.test , verbose = 0)

y.tab <- table(y.train)
y.baseline <- as.integer(names(y.tab[which.max(y.tab)]))
mean(y.test == y.baseline)



######## Validation Splits ############



