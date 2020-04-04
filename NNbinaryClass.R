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
X.train.a <- array(X.train.mat, dim(X.train.mat))
X.test.a <- array(X.test.mat, dim(X.test.mat))

y.train <- y[is.train]
y.test <- y[is.test]

library(keras)

BinaryClassification <- function(X.mat, y.vec, hiddenUnits, validationRatio, numEpochs, instruction){
  nueral.net.model <- keras_model_sequential() %>%
    layer_flatten(input_shape = ncol(X.mat)) %>% 
    layer_dense(units = hiddenUnits, activation = "sigmoid", use_bias=FALSE) %>%
    layer_dense(1, activation = "sigmoid", use_bias=FALSE)
  
  nueral.net.model %>%
    compile(
      loss = "binary_crossentropy",
      optimizer = "sgd",
      metrics = "accuracy"
    )
  
  nn.model.result <- nueral.net.model %>%
    fit(
      x = X.train.mat, y = y.train,
      epochs = numEpochs,
      validation_split = validationRatio,
      verbose = 2
    )
  
  if(instruction == "model"){
    return(nueral.net.model)
  }
  
  else{
    return(nn.model.result)
  }
}

SortData <- function(result){
  nn.metrics <- do.call(data.table::data.table, result$metrics)
  nn.metrics[, epoch := 1:.N]
  
  res.metrics.list <- list()
  res.metrics.list <- data.table::data.table(nn.metrics)
  
  res.means <- res.metrics.list[, .(
    mean.val.loss=mean(val_loss)
  ), by=epoch]
  
  return(res.means)
}

ResultsMin <- function(res.means){
  result.min.dt <- res.means[which.min(mean.val.loss)]
  result.min.dt[, point := "min"]
  return(result.min.dt)
}

result.nn.one <- BinaryClassification(X.train.mat, y.train, 10, 0.4, 100, "result")
result.nn.two <- BinaryClassification(X.train.mat, y.train, 100, 0.4, 100, "result")
result.nn.three <- BinaryClassification(X.train.mat, y.train, 1000, 0.4, 100, "result")

str(result.one)
plot(result.nn.one)
par(new=TRUE)
plot(result.two,col="darkblue")
par(new=TRUE)
plot(result.three,col="black")

nn.one.means <- SortData(result.nn.one)
nn.two.means <- SortData(result.nn.two)
nn.three.means <- SortData(result.nn.three)

str(result.nn.one$metrics)

resone.min.dt <- ResultsMin(nn.one.means)
restwo.min.dt <- ResultsMin(nn.two.means)
resthree.min.dt <- ResultsMin(nn.three.means)
str(resone.min.dt)

str(min.dt)

library(ggplot2)
ggplot()+
  geom_ribbon(aes(
    x=epoch, ymin=mean.val.loss, ymax=mean.val.loss),
    alpha=0.5,
    data=nn.one.means)+
  geom_point(aes(
    x=epoch, y=mean.val.loss),
    data=nn.one.means)+
  geom_point(aes(
    x=epoch, y=mean.val.loss, color=point),
    data=resone.min.dt)

model.nn.one <- BinaryClassification(X.train.mat, y.train, 10, 0, resone.min.dt$epoch, "model")
model.nn.two <- BinaryClassification(X.train.mat, y.train, 100, 0, restwo.min.dt$epoch, "model")
model.nn.three <- BinaryClassification(X.train.mat, y.train, 1000, 0, resthree.min.dt$epoch, "model")

model.nn.one %>%
  evaluate(X.test.mat, y.test, verbose = 0)
model.nn.two %>%
  evaluate(X.test.mat, y.test, verbose = 0)
model.nn.three %>%
  evaluate(X.test.mat, y.test, verbose = 0)

y.tab <- table(y.train)
y.baseline <- as.integer(names(y.tab[which.max(y.tab)]))
mean(y.test == y.baseline)
