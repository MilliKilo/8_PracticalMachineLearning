8_Practical Machine Learning Project
==========================

Data Set
---

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.  We are required to assess how well barbell are performed by subjects equiped with accelerometers on the belt, arm, forearm, and dumbells.

Executive Summary
The data set has been split into 2 parts:
•training (70%)
•testing (30%)

Several machine learning algorithms were tried out:
•Naive Bayes
•Linear Discrimant Analysis
•Support Vector Machines
•Adaboost
•Random Forests

Random Forest technique is followed for this project.  The basic random forest approach is improved by eliminating the least important variables: out of 52 suited variables we keep 36 with a slight increase of the accuracy in the prediction. 

Finally we double check the estimation of the out of sample error rate from cross validation and the out-of-bag error rate provided in "Random Forests" by applying the final algorithm to the test set.

The out of sample error rate is 0.5 %.

Loading and Basic Feature Selection
The data set contains two types of records:
•opening window record that contains some aggregate information
•single instant measures

We will investigate the single measures and ignore the time series measures.

library(caret)
library(randomForest)

data <- read.csv("~/Downloads/pml-training.csv")
detail <- data[data$new_window == "no",]
set.seed(1208)
inTrain<- createDataPartition(y=detail$classe, p=.7,list=F)
Training <-detail[inTrain,]
Testing<-detail[-inTrain,]

# eliminate the features that are completed NA
predictors <- sapply(1:160, function(x) all(!is.na(Training[,x]) & is.numeric(Training[,x])))
# exclude the first columns not suited as predictors
predictors[1:7] <- FALSE
num.predictors <- sum(predictors)
TrainingNoNA <- Training[,predictors]
# center and scale the features
prepro <- preProcess(TrainingNoNA, method=c("center", "scale"))
TrainingCenSca <- predict(prepro, TrainingNoNA)
# add the outcome variable
TrainingCenSca[,"classe"] <- Training[,"classe"]

 r num.predictors  predictors are kept. TrainingNoNA contains only these features.

The First Iteration of the Random Forest Classifier
# Use randomForest package to predict the classes
modelFitRf <- randomForest(classe ~ ., data = TrainingCenSca, importance=TRUE)
modelFitRf

The grown forest consists in 500 trees; at each split a sample of $7 \approx \sqrt 52$ features were tried out. Indeed each grown tree is built on the basis of a bootstrap sample. Hence on average $(1 - 1/e) * trainingSetSize$ observations are used for the tree. The remaining observations can be used to assess the prediction accuracy: this error rate is called the ouf-of-bag (OOB) error rate. Here it has been estimated to $0.56 \%$ in this basic model.

Elimination of the Least Significant Features
The importance of the features can be assessed using the varImp function. 

varImpPlot(modelFitRf)
vi <- (varImp(modelFitRf))
# considering the median. 
i <- 50
threshold <- apply(vi, 2, function(x) quantile(x, probs = i / 100))
keep.var <- apply(vi, 1, function(x) any(x > threshold))
predictors2 <- rownames(vi)[!is.na(keep.var) & keep.var == TRUE]
TrainingSel <- TrainingCenSca[,predictors2]
# add the outcome variable
TrainingSel[,"classe"] <- Training[,"classe"]
modelFitRfSel <- randomForest(classe ~ ., data = TrainingSel, importance=TRUE)
modelFitRfSel

The plots depict the relative importance of the 30 most influent features. The importance of the features is computed with the  varImp  function and, if for some class the importance of a given feature is greater than the median importance for a given class, we keep this feature. So we end with  r length(predictors2)  features. The out-of-bag error rate has fallen to 0.49 % (a $ r (.56 - .49)/.56 * 100  \%$-improvement). We need to investigate more.

The main advantage of this simplification is that we have low number of features.

The function  rfcv  computes the out of sample error rate by using cross validation when less and less variables are used.

library(ggplot2)
set.seed(1314)
res <- rfcv(TrainingSel[,-dim(TrainingSel)[2]], TrainingSel[,"classe"], cv.fold = 10)
qplot(res$n.var, res$error.cv, xlab="Number of Variables", ylab="Error Rate", main="Cross Validation Error Rate by Number of Variables", geom="line", ylim = c(0,0.1))

library(cvTools)
set.seed(1208)
K <- 10
R <- 1
errorRates <- rep(NA,K)
folds <- cvFolds(dim(TrainingSel)[1], K = K, R = R)
for (i in 1:K) {
  TestingIndex <- folds$subsets[folds$which == i]
  model <- randomForest(classe ~ ., data=TrainingSel[-TestingIndex,])
  res <- predict(model,TrainingSel[TestingIndex,])
  errorRates[i] <- sum(res != TrainingSel[TestingIndex,"classe"]) / length(res)
}

The mean out-of-sample error rate over the 10 folds is  r mean(errorRates) * 100  %. This is slightly higher than the bagging error rate.

The Final Test

TestingNoNA <- Testing[,predictors]
TestingCenSca <- predict(prepro,TestingNoNA)
TestingCenSca[,"classe"] <- Testing[,"classe"]
# check the handcrafted model
ClassesPredicted <- predict(modelFitRfSel, TestingCenSca)
confusionMatrix(ClassesPredicted, TestingCenSca[,"classe"])

For the model modelFitRfSel the out-of-bag error rate 0.49 % was remarkably close to the error rate on the test set 0.5 %.

On this data set the out-of-bag estimate of the error rate is quite reliable.
