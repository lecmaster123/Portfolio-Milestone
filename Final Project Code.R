#Alec Arroyo
#Final Project

library(arules)
library(arulesViz)
library(datasets)
library(rpart)
library(rpart.plot)
library(caret)
library(sqldf)
library(e1071)
library(class)
library(randomForest)

newmus <- read.csv("/Users/alec_arroyo/Downloads/Data/features_30_sec.csv")

#remove nulls
newmus <- na.omit(newmus)

str(newmus)

#Convert filename category into a factor
newmus$filename <- factor(newmus$filename)

#Convert label category into a factor
newmus$label <- factor(newmus$label)

colnames(newmus)

#-----------------------------------------
#Get info abt data

#Get a count of each type of label using sql statement
sqlcode <- sqldf('select COUNT(*) as Freqency, label FROM newmus group by label')

#plot bar plot
plotit <- ggplot(sqlcode, aes(x=label, y=Freqency)) +
          geom_bar(stat="identity")

#we see there is an equal dist of each of our categorical variables and how 
#they're represented in this dataset
plotit

#--------------CLUSTERING using kmeans()-------------------------#

editmusic <- na.omit(newmus)

str(editmusic)

rownames(editmusic) <- editmusic[,1]
editmusic[,1] <- NULL
editmusic <- editmusic[,c(1:58)]
#Need to change the data type for this attribute so it can calculate the means
editmusic$length <- as.numeric(editmusic$length)

#use kmeans clustering
Clusters1 <- kmeans(editmusic, 10)

#add it back to orginal dataset
newmus$Clusters <- as.factor(Clusters1$cluster)
str(editmusic$Clusters)

#plot a bar graph showing the clustering for each genre type
ggplot(data=newmus, aes(x=label, fill=Clusters))+
geom_bar(stat="count") +
labs(title = "K = ?") +
theme(plot.title = element_text(hjust=0.5), text=element_text(size=15))


#--------------Decision Tree Classifier-------------------------#

#lets remove 1/3 of the label values for the data so we can cluster 
#and see what the result is

newmus <- read.csv("/Users/alec_arroyo/Downloads/Data/features_30_sec.csv")

nolabelmusic <- na.omit(newmus)

rownames(nolabelmusic) <- nolabelmusic[,1]
nolabelmusic[,1] <- NULL
#Need to change the data type for this attribute so it can calculate the means
nolabelmusic$length <- as.numeric(nolabelmusic$length)

#train and test datasets THIS IS USING HOLDOUT METHOD------------*******

#randomize order of rows so its different genre every time
#set.seed(62)
noOrder <- sample(nrow(nolabelmusic))
nolabelmusic <- nolabelmusic[noOrder,]

#create test and train data
train <- data.frame(nolabelmusic[1,])
test <- data.frame(nolabelmusic[671,])

for(i in 1:1000) {
  if(i <= 670)
  {
    train <- rbind(train, nolabelmusic[i,])
  }
  else
  {
    test <- rbind(test, nolabelmusic[i,])
  }
}

train <- train[-1,]
test <- test[-1,]

#-----Lets see the dist of representation of our label variable on training dataset---#
sqltrain <- sqldf('select COUNT(*) as Freqency, label FROM train group by label')

plotit <- ggplot(sqltrain, aes(x=label, y=Freqency)) +
  geom_bar(stat="identity")

#we see there is an equal dist of each of our categorical variables and how 
#they're represented in this dataset
plotit


decision <- rpart(label~., train, method="class")

predictionnotouch <- predict(decision, test, type="class")

#needs to be a factor for Confusion Matrix to work
test$label <- as.factor(test$label)

#work on this get prediction accuracy
confusionMatrix(predictionnotouch, test$label)

#maybe dont use unknown value and see if it can accurately predict all teh known genre files
#Try training you're dataset using cross validation instead of 3/2 split

#-----------------------------KFOLDS---------------------------------
#Since this is a small dataset of 1000, we want to use something that will provide more 
#variety in the training data to represent everything in the dataset
#make sure you talk about using hldout vs cross validation for my dataset and why cross
#validation is important

#numrow <- nrow(nolabelmusic)
kfolds <- 6

#crossval <- split(sample(1:numrow), 1:kfolds)

sumAll<-list()
sumLabel<-list()
nolabelmusicCVAL <- nolabelmusic

for(k in 1:kfolds) {
  
  noOrderCV <- sample(nrow(nolabelmusicCVAL))
  nolabelmusicCVAL <- nolabelmusicCVAL[noOrderCV,]
  
  trainCV <- data.frame(nolabelmusicCVAL[1,])
  testCV <- data.frame(nolabelmusicCVAL[671,])
  
  for(i in 1:1000) {
    if(i <= 670)
    {
      trainCV <- rbind(trainCV, nolabelmusicCVAL[i,])
    }
    else
    {
      testCV <- rbind(testCV, nolabelmusicCVAL[i,])
    }
  }
  
  trainCV <- trainCV[-1,]
  testCV <- testCV[-1,]
  
  Dtreetrain <- trainCV
  Dtreetest <- testCV
  
  Dtreetest$label <- as.factor(Dtreetest$label)
  
  train_Dtree <- rpart(label~., Dtreetrain, method="class")
  nb_Pred <- predict(train_Dtree, Dtreetest, type="class")
  
  sumAll <- c(sumAll,nb_Pred)
  sumLabel <- c(sumLabel, Dtreetest$label)
  
}

#-----Lets see the dist of representation of our label variable on cross validation dataset---# 
sqltrainCV <- sqldf('select COUNT(*) as Freqency, label FROM Dtreetrain group by label')

plotit <- ggplot(sqltrainCV, aes(x=label, y=Freqency)) +
  geom_bar(stat="identity")

#we see there is an equal dist of each of our categorical variables and how 
#they're represented in this dataset
plotit

confusionMatrix(nb_Pred, Dtreetest$label)

table(unlist(sumAll), unlist(sumLabel))

get_accuracy_rate <- function(results_table, label) { 
  diagonal_sum <- sum(c(results_table[[1]], results_table[[12]], results_table[[23]], results_table[[34]], 
                        results_table[[45]], results_table[[56]], results_table[[67]], results_table[[78]], 
                        results_table[[89]], results_table[[100]])) 
  (diagonal_sum / label)*100 
}

all_results <- data.frame(orig=unlist(sumAll), pred=unlist(sumLabel))

get_accuracy_rate(table(all_results$orig, all_results$pred), length(all_results$pred)) 


#use kfolds method to try on decision tree and see if it increases the accuracy
#So far Decision Tree winning with 59% accuracy and Bayes in second with 47% accuracy

#----------__-----RANDOM FOREST------__--------------------------#

#Random Forest classifier on training and testing data
rfmusic <- train(label ~ ., train, method="rf")
rf_Pred <- predict(rfmusic, test)
test$label <- as.factor(test$label)
confusionMatrix(rf_Pred, test$label)
