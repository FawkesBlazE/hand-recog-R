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