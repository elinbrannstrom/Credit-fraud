#Loading libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)

# Link to kaggle data set: https://www.kaggle.com/ntnu-testimon/paysim1
# setwd("C:/Users/ebrannstrom/Downloads")
# credit_fraud <- read.csv("credit_fraud.csv")
credit_fraud <- read.csv("https://www.dropbox.com/s/mz93hln4efohpn4/credit_fraud.csv?dl=1")
head(credit_fraud)

#Dimensions of full dataset

dim(credit_fraud)

#Examine rows flagged as fraud, and conclude these are very few and can be deleted
nrow(credit_fraud[credit_fraud$isFlaggedFraud==1,])
nrow(credit_fraud[credit_fraud$isFlaggedFraud==1,]) / nrow(credit_fraud)

#Remove isFlaggedFraud-column
credit_fraud_new <- subset(credit_fraud, select=-isFlaggedFraud)

#Check for missing values
sum(is.na(credit_fraud_new))

#Checking amounts of fraud and conclude small amounts of fraud are most common
credit_fraud_new %>% filter(isFraud==1) %>% ggplot(aes(amount)) +geom_histogram(bins=20)

#Checking number of unique steps & plotting time dependency
n_distinct(credit_fraud_new$step)

credit_fraud_new %>% group_by(step) %>% summarize(rating=mean(isFraud)) %>% 
  ggplot(aes(step,rating)) + geom_point() + geom_smooth()

credit_fraud_new %>% group_by(step) %>% summarize(n=n()) %>% 
  ggplot(aes(step,n)) + geom_point() + geom_smooth()

#Checking number of unique values for some columns containing text, 99.8 % vs 42.7 % are unique, won't be helpful for modelling
# No fraud amount larger than 10 000 000
nr_unique_nameOrig <- length(unique(credit_fraud_new$nameOrig))/dim(credit_fraud)[1]
nr_unique_nameDest <- length(unique(credit_fraud_new$nameDest))/dim(credit_fraud)[1]

#Replace with first letter of rows
credit_fraud_new <- credit_fraud_new %>% mutate(nameOrigLetter = str_sub(nameOrig,1,1)) %>% mutate(nameDestLetter = str_sub(nameDest,1,1)) %>%
  select(-nameOrig,-nameDest)

length(unique(credit_fraud_new$nameOrigLetter))
credit_fraud_new <- credit_fraud_new %>% select(-nameOrigLetter)

length(unique(credit_fraud_new$nameDestLetter))


#Check correlation between numeric columns. Notice strong correlation between oldbalanceDest & newbalanceDest and newbalanceOrig&oldbalanceOrig
test_corr <- credit_fraud_new %>% select(-nameOrigLetter,-nameDestLetter,-type)
cor(test_corr)

# Find which columns to remove
findCorrelation(cor(test_corr),cutoff = .7, verbose = TRUE, names = TRUE, exact = TRUE)

# Remove columns
credit_fraud_new <- credit_fraud_new %>% select(-newbalanceOrig,-newbalanceDest)

# Exmine the fraudulent cases. Notice how fraud only occurs for the types "CASH_OUT" & "TRANSFER".
# Fraud also only occurs for "C" as "nameDestLetter" and at a maximum of 10 million. 
# This will help us reduce the data set.
only_fraud <- credit_fraud_new %>% filter(isFraud==1)
summary(only_fraud)

# Filter data set based on summary of fraudulent cases

credit_fraud_new <- credit_fraud_new %>% filter(nameDestLetter == "C") %>%
  filter(amount <= 10000000) %>% filter(type %in% c("CASH_OUT", "TRANSFER")) %>%
  select(-nameDestLetter)

# Size of reduced data set compared to original data set
dim(credit_fraud_new)[1]/dim(credit_fraud)[1]

#Partitioning data - APPROACHED NOT USED IN FINAL MODEL DUE TO COMPUTATIONAL CONSTRAINTS
credit_fraud_new$type <- as.factor(credit_fraud_new$type)
credit_fraud_new$isFraud <- as.factor(credit_fraud_new$isFraud)
set.seed(1,sample.kind="Rounding")

test_index <- createDataPartition(y = credit_fraud_new$isFraud, times=1, p=0.1, list=FALSE)
validation_set <- credit_fraud_new[test_index,]
test_set <- credit_fraud_new[-test_index,]

set.seed(1,sample.kind="Rounding")
test_index <- createDataPartition(y = test_set$isFraud, times=1, p=0.2, list=FALSE)
training_set <- credit_fraud_new[-test_index,]
test_set <- credit_fraud_new[test_index,]

# Tried training a knn model, however did not terminate within 24 hours on laptop,
#so a different approach needs to be considered
knn_model = train(isFraud ~ .,
                    data = training_set,
                    method = "knn")

# UNDER-SAMPLING to reduce data

number_of_frauds <- sum(credit_fraud_new$isFraud==1)
only_fraud <- credit_fraud_new[which(credit_fraud_new$isFraud==1),]
only_no_fraud <- credit_fraud_new[which(credit_fraud_new$isFraud==0),]

set.seed(1,sample.kind="Rounding")
under_sampling_no_fraud <- sample_n(only_no_fraud, number_of_frauds)
under_sampled_set <- bind_rows(under_sampling_no_fraud, only_fraud)

# Replacing data set with under sampled one. Overwriting the name to be able to reuse code
credit_fraud_new <- under_sampled_set


#Changing integers to factors
credit_fraud_new$type <- as.factor(credit_fraud_new$type)
credit_fraud_new$isFraud <- as.factor(credit_fraud_new$isFraud)

#Set seed for reproducibility
set.seed(1,sample.kind="Rounding")

# Create data partition from under smapled set, using 10 % for validation,
#and 20 % of the remaining data for testing. 
#Remaining part is used for training.
test_index <- createDataPartition(y = credit_fraud_new$isFraud, times=1, p=0.1, list=FALSE)
validation_set <- credit_fraud_new[test_index,]
test_set <- credit_fraud_new[-test_index,]

set.seed(1,sample.kind="Rounding")
test_index <- createDataPartition(y = test_set$isFraud, times=1, p=0.2, list=FALSE)
training_set <- credit_fraud_new[-test_index,]
test_set <- credit_fraud_new[test_index,]

#MODELLING

#Accuracy when predicting all transactions in original data as legal
fraud_as_numeric <- as.numeric(as.character(credit_fraud$isFraud))
accuracy_all_legal <- 1 - sum(fraud_as_numeric)/dim(credit_fraud)[1]
print(accuracy_all_legal)

#Control parameter for modelling
control <- trainControl(method = "cv", number = 10, p = .9)

#Training glm model
train_glm <- train(isFraud ~ ., method = "glm", 
                   data = training_set,
                   trControl = control)

#Predict onto test set and display confusion matrix and accuracy
glm_pred <- predict(train_glm, test_set)
cm_glm <- confusionMatrix(glm_pred, test_set$isFraud)

print(cm_glm)
glm_accuracy <- cm_glm$overall['Accuracy']

accuracy_results <- data_frame(method= "GLM", ACC=glm_accuracy)
accuracy_results %>% knitr::kable()

#Train knn model
train_knn <- train(isFraud ~ ., method = "knn", 
                   data = training_set,
                   trControl = control)

#Predict onto test set and display confusion matrix and accuracy
knn_pred <- predict(train_knn, test_set)
cm_knn <- confusionMatrix(knn_pred, test_set$isFraud)

print(cm_knn)

knn_accuracy <- cm_knn$overall['Accuracy']

accuracy_results <- bind_rows(accuracy_results, tibble(method="KNN", ACC=knn_accuracy))
accuracy_results %>% knitr::kable()

#Train Random Forest model
train_rf <- train(isFraud ~ ., method = "rf", 
                  data = training_set,
                  trControl = control)

#Predict onto test set and display confusion matrix
rf_pred <- predict(train_rf, test_set)
cm_rf <- confusionMatrix(rf_pred, test_set$isFraud)

print(cm_rf)

rf_accuracy <- cm_rf$overall['Accuracy']

accuracy_results <- bind_rows(accuracy_results, tibble(method="RF", ACC=rf_accuracy))
accuracy_results %>% knitr::kable()

#Train Support Vector Machine model
train_svm <- train(isFraud ~ ., method = "svmRadial", 
                   data = training_set,
                   trControl = control)

#Predict onto test set and display confusion matrix
svm_pred <- predict(train_svm, test_set)
cm_svm <- confusionMatrix(svm_pred, test_set$isFraud)

print(cm_svm)

svm_accuracy <- cm_svm$overall['Accuracy']

accuracy_results <- bind_rows(accuracy_results, tibble(method="SVM", ACC=svm_accuracy))
accuracy_results %>% knitr::kable()

#Continue working with the Random Forest model which showed the best results
#when modelling. Introducing cross-validation for the mtry parameter

#Define tuning grid for the mtry parameter
tunegrid <- expand.grid(.mtry=c(1:5))
#Train Random Forest model
train_rf_crossv <- train(isFraud ~ ., method = "rf", 
                         tuneGrid = tunegrid, 
                         data = training_set,
                         trControl = control)

#Predict onto test set and display confusion matrix
rf_crossv_pred <- predict(train_rf_crossv, test_set)
cm_rf_crossv <- confusionMatrix(rf_crossv_pred, test_set$isFraud)

print(cm_rf_crossv)

rf_crossv_accuracy <- cm_rf_crossv$overall['Accuracy']

accuracy_results <- bind_rows(accuracy_results, tibble(method="RF crossv", ACC=rf_crossv_accuracy))
accuracy_results %>% knitr::kable()

#Using the Random Forest model which showed highest accuracy to predict onto validation data
validation_pred <- predict(train_rf, validation_set)
cm_validation <- confusionMatrix(validation_pred, validation_set$isFraud)

print(cm_validation)

validation_accuracy <- cm_validation$overall['Accuracy']
accuracy_results <- bind_rows(accuracy_results, tibble(method="VALID.", ACC=validation_accuracy))
accuracy_results %>% knitr::kable()

#Predict on a sampled subset of legal transactions from the subset where fraud
#can occur
set.seed(1,sample.kind="Rounding")
subset_no_fraud <- sample_n(only_no_fraud, dim(under_sampled_set)[1])
subset_no_fraud$type <- as.factor(subset_no_fraud$type)
subset_no_fraud$isFraud <- as.factor(subset_no_fraud$isFraud)

#Predict using the random forest model and calculate accuracy
no_fraud_pred <- predict(train_rf, subset_no_fraud)
no_fraud_accuracy <- 1 - sum(no_fraud_pred==1)/dim(subset_no_fraud)[1]
accuracy_results <- bind_rows(accuracy_results, tibble(method="NO FRAUD", ACC=no_fraud_accuracy))
accuracy_results %>% knitr::kable()