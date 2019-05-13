data <- read.csv('digit-recognizer\\train.csv')

sample <- matrix(as.numeric(data[4,-1]),nrow = 28, byrow = TRUE)
image(sample,col = grey.colors(255))

rotate <- function(x) t(apply(x,2,rev))
image(rotate(sample),col=grey.colors(255))


data$label <- as.factor(data$label)


central_block <- c("pixel376", "pixel377", "pixel404", "pixel405")
par(mfrow=c(2,2))
for(i in 1:9){
  hist(c(as.matrix(data[data$label==i,central_block])), 
       main=sprintf("Histogram for digit %d", i), xlab="Pixel value")
}


library(caret)
set.seed(42)
train_p=0.75
train_index <- createDataPartition(data$label, p=train_p, list=FALSE)
data_train <- data[train_index,]
data_test <- data[-train_index,]

library(nnet)
model_nn <- nnet(label ~ ., data=data_train, size=50, MaxNWts=100000, decay=1e-3, maxit=100)
prediction_nn <- predict(model_nn, data_test, type="class")
cm_nn = table(data_test$label, prediction_nn)
cm_nn
accuracy_nn = mean(prediction_nn == data_test$label)
accuracy_nn


cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/"
options(repos = cran)

if(!require("mxnet"))
  install.packages("mxnet")

require(mxnet)
data_train <- data.matrix(data_train)
data_train.x <- data_train2[,-1]
data_train.x <- t(data_train2.x/255)
data_train.y <- data_train2[,1]

data <- mx.symbol.Variable("data")
fc1 <- mx.symbol.FullyConnected(data, name ="fc1", num_hidden = 128)
act1 <- mx.symbol.Activation(fc1, name="relu1", act_type = "relu")
fc2 <- mx.symbol.FullyConnected(act1, name ="fc2", num_hidden = 64)
act2 <- mx.symbol.Activation(fc2, name="relu2", act_type = "relu")
fc3 <- mx.symbol.FullyConnected(act2, name ="fc3", num_hidden = 10)
softmax <- mx.symbol.SoftmaxOutput(fc3,name="sm")
devices <- mx.cpu()
mx.set.seed(42)
model_dnn <- mx.model.FeedForward.create(softmax, x=data_train.x, y=data_train.y, ctx=devices, 
                                         num.round = 30, array.batch.size = 100, learning.rate=0.01, 
                                         momentum=0.9, eval.metric=mx.metric.accuracy, 
                                         initializer = mx.init.uniform(0.1),
                                         epoch.end.callback = mx.callback.log.train.metric(100))
data_test.x <- data_test[,-1]
data_test.x <- t(data_test.x/255)

prob_dnn <- predict(model_dnn, data_test.x)
prediction_dnn <- max.col(t(prob_dnn)) -1
cm_dnn = table(data_test$label, prediction_dnn)
cm_dnn
accuracy_dnn = mean(prediction_dnn == data_test$label)
accuracy_dnn

