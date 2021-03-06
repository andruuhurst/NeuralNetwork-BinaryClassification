#### install tensorflow ######
install.packages("tensorflow")

library(tensorflow)
install_tensorflow()

#### install keras ######
install.packages("keras")

library(keras)

#### download spam data set #####
if(!file.exists("spam.data")){
  download.file(
    "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/spam.data",
    "spam.data")
}

spam.dt <- data.table::fread("spam.data")

label.col <- ncol(spam.dt)
y <- array(spam.dt[[label.col]], nrow(spam.dt))

set.seed(1)
fold.vec <- sample(rep(1:5, l=nrow(spam.dt)))
test.fold <- 1
is.test <- fold.vec == test.fold
is.train <- !is.test

X.sc <- scale(spam.dt[, -label.col, with=FALSE])

X.train.mat <- X.sc[is.train,]
X.test.mat <- X.sc[is.test,]
X.train.arr <- array(X.train.mat , dim(X.train.mat))
X.test.arr <- array(X.test.mat , dim(X.test.mat))

y.train <- y[is.train]
y.test <- y[is.test]

######## Neural Network : n.hidden_units ######

hidden.unit.metrics.list <- list()
for( n.hidden.units in c( 10 , 100 , 1000)){
      
    model <- keras_model_sequential() %>%
      layer_flatten( input_shape = ncol(X.train.mat))  %>%                                  # input layer
      layer_dense(units = n.hidden.units , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
      layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )                     # ouput layer
    
    model %>% 
      compile(
        loss = "binary_crossentropy",
        optimizer = "sgd",
        metrics = "accuracy"
      )
    
    result <- model %>% 
      fit(
        x = X.train.mat, y = y.train,
        epochs = 100,
        validation_split = 0.4,
        verbose = 2
      )
    
    unit.metrics <- do.call(data.table::data.table , result$metrics)
    unit.metrics[, epoch := 1:.N]
    hidden.unit.metrics.list[[length(hidden.unit.metrics.list)+1]] <- data.table::data.table(
     n.hidden.units, unit.metrics )
}

hidden.unit.metrics <- do.call(rbind, hidden.unit.metrics.list)


#### find min epochs for loss and val_loss #####

## log loss
min.loss.dt.list <-list()
uni.unit.vec <- unique(hidden.unit.metrics[, n.hidden.units])
for( units in uni.unit.vec){
  min.loss.dt.list[[length(min.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(loss)]
}

min.loss.dt <- do.call(rbind, min.loss.dt.list)


## val log loss
min.val.loss.dt.list <-list()
for( units in uni.unit.vec){
  min.val.loss.dt.list[[length(min.val.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(val_loss)]
}

min.val.loss.dt <- do.call(rbind, min.val.loss.dt.list)



###### plot : log loss / epoch #######
library(ggplot2)

ggplot()+
  geom_line( aes( x= epoch,
                  y= loss ,
                  color= n.hidden.units,
                  group = n.hidden.units),
             size=1,
             data = hidden.unit.metrics) +
  geom_point( aes( x= epoch,
                   y= loss ),
              color = "red",
              size= 1.5,
              min.loss.dt)

ggplot()+
  geom_line( aes( x= epoch,
                  y= val_loss,
                  color= n.hidden.units,
                  group= n.hidden.units),
             data = hidden.unit.metrics)+
  geom_point( aes( x= epoch,
                   y= val_loss ),
              color = "red",
              size= 1.5,
              min.val.loss.dt)


#### find min epochs for loss and val_loss #####

## log loss
min.loss.dt.list <-list()
uni.unit.vec <- unique(hidden.unit.metrics[, n.hidden.units])
for( units in uni.unit.vec){
  min.loss.dt.list[[length(min.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(loss)]
}

min.loss.dt <- do.call(rbind, min.loss.dt.list)


## val log loss
min.val.loss.dt.list <-list()
for( units in uni.unit.vec){
  min.val.loss.dt.list[[length(min.val.loss.dt.list)+1]] <- hidden.unit.metrics[ n.hidden.units == units][which.min(val_loss)]
}

min.val.loss.dt <- do.call(rbind, min.val.loss.dt.list)

######## Re-train model with best epochs ##########

 train.min.dt.list <- list()

  #### 10 hidden units with best epoch : 100 #####
  model <- keras_model_sequential() %>%
    layer_flatten( input_shape = ncol(X.train.mat))  %>%                                  # input layer
    layer_dense(units = 10 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
    layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )                     # ouput layer
  
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  result1 <- model %>% 
    fit(
      x = X.train.mat, y = y.train,
      epochs = 100,
      validation_split = 0,
      verbose = 2
    )
  
  unit.metrics <- do.call(data.table::data.table , result1$metrics)
  unit.metrics[, epoch := 1:.N]
  train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]
  
  #### 100 hidden units with best epoch : 85 #####
  model <- keras_model_sequential() %>%
    layer_flatten( input_shape = ncol(X.train.mat))  %>%                                  # input layer
    layer_dense(units = 100 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
    layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )                     # ouput layer
  
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  result2 <- model %>% 
    fit(
      x = X.train.mat, y = y.train,
      epochs = 85,
      validation_split = 0,
      verbose = 2
    )
  unit.metrics <- do.call(data.table::data.table , result2$metrics)
  unit.metrics[, epoch := 1:.N]
  train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]
  
  #### 1000 hidden units with best epoch : 37 #####
  model <- keras_model_sequential() %>%
    layer_flatten( input_shape = ncol(X.train.mat))  %>%                                  # input layer
    layer_dense(units = 1000 , activation = "sigmoid" , use_bias = FALSE ) %>%  # hidden layer
    layer_dense(units= 1 , activation = "sigmoid", use_bias = FALSE )                     # ouput layer
  
  model %>% 
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  result3 <- model %>% 
    fit(
      x = X.train.mat, y = y.train,
      epochs = 37,
      validation_split = 0,
      verbose = 2
    )
  unit.metrics <- do.call(data.table::data.table , result3$metrics)
  unit.metrics[, epoch := 1:.N]
  train.min.dt.list[[length(train.min.dt.list)+1]] <- unit.metrics[which.min(loss)]
  
  ###### Baseline prediction ######
  y.tab <- table(y.train)
  y.baseline <- as.integer(names(y.tab[which.max(y.tab)]))
  mean(y.test == y.baseline)
  
  
  
  