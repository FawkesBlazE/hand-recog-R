# first convolution
conv1 <- mx.symbol.Convolution(data=data, kernel=c(5,5),num_filter=20)
act1 <- mx.symbol.Activation(data=conv1, act_type="relu")
pool1 <- mx.symbol.Pooling(data=act1, pool_type="max", kernel=c(2,2), stride=c(2,2))

# second convolution
conv2 <- mx.symbol.Convolution(data=pool1, kernel=c(5,5),num_filter=50)
act2 <- mx.symbol.Activation(data=conv2, act_type="relu")
pool2 <- mx.symbol.Pooling(data=act2, pool_type="max",kernel=c(2,2), stride=c(2,2))

flatten <- mx.symbol.Flatten(data=pool2)

# first fully connected layer
fc1 <- mx.symbol.FullyConnected(data=flatten, num_hidden=500)
act3 <- mx.symbol.Activation(data=fc1, act_type="relu")
# second fully connected layer
fc2 <- mx.symbol.FullyConnected(data=act3, num_hidden=10)

# softmax output
softmax <- mx.symbol.SoftmaxOutput(data=fc2, name="sm")

train.array <- data_train.x
dim(train.array) <- c(28, 28, 1, ncol(data_train.x))
mx.set.seed(42)
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array,
                                         y=data_train.y, ctx=devices, num.round=30,
                                         array.batch.size=100, learning.rate=0.05,momentum=0.9, 
                                         wd=0.00001,
                                         eval.metric=mx.metric.accuracy,
                                         epoch.end.callback=mx.callback.log.train.metric(100))

test.array <- data_test.x
dim(test.array) <- c(28, 28, 1, ncol(data_test.x))

prob_cnn <- predict(model_cnn, test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_test$label, prediction_cnn)
cm_cnn
accuracy_cnn = mean(prediction_cnn == data_test$label)
accuracy_cnn
graph.viz(model_cnn$symbol)

data_test.y <- data_test[,1]
logger <- mx.metric.logger$new()
model_cnn <- mx.model.FeedForward.create(softmax, X=train.array,
                                         y=data_train.y,eval.data=list(data=test.array,
                                          label=data_test.y), ctx=devices, num.round=30,
                                           array.batch.size=100, learning.rate=0.05,
                                           momentum=0.9, wd=0.00001,eval.metric=
                                        mx.metric.accuracy, epoch.end.callback = mx.callback.log.train.metric(1, logger))

logger$train
logger$eval
plot(logger$train,type="l",col="red", ann=FALSE)
lines(logger$eval,type="l", col="blue")
title(main="Learning curve")
title(xlab="Iterations")
title(ylab="Accuary")
legend(20, 0.5, c("training","testing"), cex=0.8,col=c("red","blue"), pch=21:22, lty=1:2);

par(mfrow=c(1,2))
test_1 <- matrix(as.numeric(data_test[1,-1]), nrow = 28, row = TRUE)
image(rotate(test_1), col = grey.colors(255))
test_2 <- matrix(as.numeric(data_test[2,-1]), nrow = 28, byrow = TRUE)
image(rotate(test_2), col = grey.colors(255))

layerss_for_viz <- mx.symbol.Group(mx.symbol.Group(c(conv1, act1, pool1, conv2, act2, pool2, fc1, fc2)))
executor <- mx.simple.bind(symbol=layerss_for_viz,
                           data=dim(test.array), ctx=mx.cpu()
                           mx.exec.update.arg.arrays(executor, model_cnn$arg.params, match.name=TRUE)
mx.exec.update.aux.arrays(executor, model_cnn$aux.params, match.name=TRUE)
mx.exec.update.arg.arrays(executor, list(data=mx.nd.array(test.array)), match.name=TRUE)
mx.exec.forward(executor, is.train=FALSE)
names(executor$ref.outputs)
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$activation15_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n',
        col=grey.colors(255) )
}
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$activation15_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
        col=grey.colors(255))
}

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <-as.array
  (executor$ref.outputs$pooling10_output)[,,i,1]
  image(outputData, xaxt='n', yaxt='n',
        col=grey.colors(255))
}
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$pooling10_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255))}

par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) { 
  outputData <- as.array(executor$ref.outputs$convolution11_output)[,,i,1]image(outputData, xaxt='n', yaxt='n',col=grey.colors(255)
 )
}
par(mfrow=c(4,4), mar=c(0.1,0.1,0.1,0.1))
for (i in 1:16) {
  outputData <- as.array
  (executor$ref.outputs$convolution11_output)[,,i,2]
  image(outputData, xaxt='n', yaxt='n',
          col=grey.colors(255)
          )
  }

validation_perc = 0.4
validation_index <- createDataPartition(data_test.y, p=validation_perc, list=FALSE)

validation.array <- test.array[, , , validation_index]
dim(validation.array) <- c(28, 28, 1,
                             length(validation.array[1,1,]))
data_validation.y <- data_test.y[validation_index]
final_test.array <- test.array[, , , -validation_index]
dim(final_test.array) <- c(28, 28, 1,
                             length(final_test.array[1,1,]))
data_final_test.y <- data_test.y[-validation_index]
mx.callback.early.stop <- function(eval.metric) {
  function(iteration, nbatch, env, verbose) {
    if (!is.null(env$metric)) {
      if (!is.null(eval.metric)) {
        result <- env$metric$get(env$eval.metric)
        if (result$value >= eval.metric) {
          return(FALSE)
          }
        }
      }
    return(TRUE)
    }
  }

model_cnn_earlystop <- mx.model.FeedForward.create(softmax,
                                                   X=train.array, y=data_train.y,
                                                   eval.data=list(data=validation.array, label=data_validation.y),
                                                   + ctx=devices, num.round=30, array.batch.size=100,
                                                   + learning.rate=0.05, momentum=0.9, wd=0.00001,
                                                   eval.metric=mx.metric.accuracy,
                                                   + epoch.end.callback = mx.callback.early.stop(0.985))
prob_cnn <- predict(model_cnn_earlystop, final_test.array)
prediction_cnn <- max.col(t(prob_cnn)) - 1
cm_cnn = table(data_final_test.y, prediction_cnn)
cm_cnn
