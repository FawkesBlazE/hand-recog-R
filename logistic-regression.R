setwd("C:/Users/Fawkes Blaze/Desktop/Python")
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
model_lr <- multinom(label ~ ., data=data_train, MaxNWts=10000, decay=5e-3, maxit=100)
prediction_lr <- predict(model_lr, data_test, type = "class")
prediction_lr [1:5]
accuracy_lr = mean(prediction_lr == data_test$label)
accuracy_lr

