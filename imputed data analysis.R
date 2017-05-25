##########################
#### Titanic ML Script ###
##########################

library(caret)
library(dplyr)
library(ggplot2)

rawData <- read.csv("./train.csv", header = TRUE, na.strings = "")
testSet <- read.csv("./test.csv", header = TRUE, na.strings = "")

testSet$Survived <- as.factor(testAnswer$Survived)

# Convert embarked location into numerical variable to ease analysis
rawData$Embarked <- as.numeric(rawData$Embarked)
testSet$Embarked <- as.numeric(testSet$Embarked)

rawData$Sex <- as.numeric(rawData$Sex)
testSet$Sex <- as.numeric(testSet$Sex)

# Convert Survived into factor variable
rawData$Survived <- as.factor(rawData$Survived)

set.seed(808)
inTraining <- createDataPartition(y = rawData$Survived, p = 0.7, list = FALSE)

training <- rawData[inTraining,]
testing <- rawData[-inTraining,]

names(training)

# Remove PassengerId, ticket, cabin, and name, as they all are unlikely to provide predictive
# information. Do this for training and test set

training <- select(training, -c(PassengerId, Ticket, Cabin, Name))
testing <- select(testing, -c(PassengerId, Ticket, Cabin, Name))

imputeTestSet <-select(testSet, -c(PassengerId, Ticket, Cabin, Name))

## We notice that age has missing values in age and embarked location. 
## We will create a new data set with imputed age and embark location

preProc <- preProcess(training[,-1], method = "knnImpute", k = 10)

imputedTraining <-predict(preProc, training)

# Use same preprocessing for test set

imputedTesting <-predict(preProc, testing)

imputeTestSet <- predict(preProc, imputeTestSet)


#################### Create Prediction Models #################################


########################### Parallel Processing ########################
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
############################ Parallel Processing  #######################


################################################# Classification Tree
library(rattle)
set.seed(808)
treeImputed <- train(Survived~., method ="rpart", data = imputedTraining)

fancyRpartPlot(treeImputed$finalModel)

# 79.33%
confusionMatrix(treeImputed)

############################################### Random Forrest
rfControl <- trainControl(allowParallel = TRUE)

set.seed(808)
system.time(rfImputed <- train(Survived~., method ="rf", data = imputedTraining, trCrontrol = rfControl))


############################################### Boosting
gbmControl <- trainControl(allowParallel = TRUE)

set.seed(808)
system.time(titanicBoost <- train(Survived~., data = imputedTraining, method = "gbm", trControl = gbmControl))
#81.42
confusionMatrix(rfImputed)
#80.84
confusionMatrix(titanicBoost)

############################################################### Combined Model

# Create predictions for data frame
rfPredict <- predict(rfImputed, imputedTesting)
boostPredict <- predict(titanicBoost, imputedTesting)
treePredict <- predict(treeImputed, imputedTesting)

combinedModData <- data.frame(rfPredict, boostPredict, treePredict, Survived =  imputedTesting$Survived)

set.seed(808)
combRFModel <- train(Survived~., data = combinedModData, method = "rf", trControl = rfControl)
set.seed(808)
combBoostModel <- train(Survived~., data = combinedModData, method = "gbm", verbose = F)

############## End parallel processing ########
stopCluster(cluster)
registerDoSEQ()
############## End parallel processing ########

### Test Set predictions

finalTreePred <- predict(treeImputed, imputeTestSet)
finalRFPred <- predict(rfImputed, imputeTestSet)
finalboostPred <- predict(titanicBoost, imputeTestSet)

finalStackedData <- data.frame(rfPredict = finalRFPred, boostPredict = finalboostPred,
                              treePredict = finalTreePred)

testSetPred <- predict(combRFModel, finalStackedData)

testSetBoostPred <- predict(combBoostModel, finalStackedData)

### Create prediction dataframe/CSV


predictionDF <- data.frame(PassengerId = testSet$PassengerId, Survived = testSetBoostPred)

predictionDfRf <- data.frame(PassengerId = testSet$PassengerId, Survived = testSetPred)

## 77.03% accuracy
write.csv(predictionDF, "./titanicPredictions1.csv", row.names = F)


write.csv(predictionDfRf, "./titanicPredictions2rf.csv", row.names = F)

