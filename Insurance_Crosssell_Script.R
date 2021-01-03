# Attaching required packages

if(!require(tidyverse))
  install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(data.table))
  install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart))
  install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(caret)) 
  install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(PRROC)) 
  install.packages("PRROC",  repos = "http://cran.us.r-project.org")
if(!require(ROCR)) 
  install.packages("ROCR", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) 
  install.packages("xgboost", repos = "http://cran.us.r-project.org")
if(!require(corrr)) 
  install.packages("corrr", repos = "http://cran.us.r-project.org")

library(corrr)

# Downloading database
# https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

train <- read_csv("./Insurance_Crosssell/train.csv")
str(train)

# Converting character variables to numeric

train$gender[train$Gender == "Male"] <- 1
train$gender[train$Gender != "Male"] <- 0
train$damage[train$Vehicle_Damage == "Yes"] <- 1
train$damage[train$Vehicle_Damage != "Yes"] <- 0
train$veh_age[train$Vehicle_Age == "> 2 Years"] <- 2
train$veh_age[train$Vehicle_Age == "< 1 Year"] <- 0
train$veh_age[train$Vehicle_Age == "1-2 Year"] <- 1

train <- train %>% select(gender, Age, Previously_Insured, veh_age, damage, Annual_Premium, Policy_Sales_Channel, Response)

str(train)

# Splitting the data into 80:20 between train and test set
set.seed(1234, sample.kind = "Rounding") 
test_index <- createDataPartition(train$Response, times = 1, p = 0.2, list = FALSE)
test_set <- train[test_index,]
train_set <- train[-test_index,]

# Use the data to determine correlation of Response with other variables.

Cor_response <- train %>% correlate() %>% focus(Response)

Cor_response

# Proportion of Yes response in training set and test set suggests that No (or 0 ) has higher occurence
mean(train_set$Response == 1)
mean(test_set$Response == 1)

# Use the training set to determine whether a given Gender were more likely to respond yes. Apply this insight to generate predictions on the test set.

Gender_response <- train_set %>% group_by(gender) %>% summarize(Response_prop = mean(Response == 1))
Gender_response

# Use the training set to determine in previously insured people were more likely to respond yes. we will Apply this insight to generate predictions on the test set.

Prev_Ins_response <- train_set %>% group_by(Previously_Insured) %>% summarize(Response_prop = mean(Response == 1))

Prev_Ins_response


# Use the training set to determine if vehicle damage leads to more response of yes. we will Apply this insight to generate predictions on the test set.

Damage_response <- train_set %>% group_by(damage) %>% summarize(Response_prop = mean(Response == 1))

Damage_response

# Cleaning the data to keep only relevant features
train_set <- train_set %>% select(Age, Previously_Insured, veh_age, damage, Policy_Sales_Channel, Response)

test_set  <- test_set  %>% select(Age, Previously_Insured, veh_age, damage, Policy_Sales_Channel, Response)

# The simplest prediction method is randomly guessing the outcome without using additional predictors. These methods will help us determine whether our machine learning algorithm performs better than chance.

set.seed(3, sample.kind = "Rounding")

guess_response <- sample(c(0,1), nrow(test_set), replace = TRUE)

## Calculating the accuracy for the Random Prediction Model

confusionMatrix(data = factor(guess_response), reference = factor(test_set$Response))

## Calculating the AUC for the Random Prediction Model

pred_guess <- prediction(
  as.numeric(as.character(guess_response)),as.numeric(as.character(test_set$Response)))

auc_val_guess <- performance(pred_guess, "auc")
auc_plot_guess <- performance(pred_guess, 'sens', 'spec')
plot(auc_plot_guess, main=paste("AUC:", auc_val_guess@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- tibble(Method = "random_prediction", Accuracy = confusionMatrix(data = factor(guess_response), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_guess@y.values[[1]])

result

# Since not having insurance previously is likely to have higher affirmative response rate, lets create a model for this and check its parameters.

Prev_Ins_model <- ifelse(test_set$Previously_Insured == 0, 1, 0)

## Calculating the accuracy for the Previous Insurance Model

confusionMatrix(data = factor(Prev_Ins_model), reference = factor(test_set$Response))

## Calculating the AUC for the Previous Insurance Model

pred_Prev_Ins <- prediction(
  as.numeric(as.character(Prev_Ins_model)),as.numeric(as.character(test_set$Response)))

auc_val_Prev_Ins <- performance(pred_Prev_Ins, "auc")
auc_plot_Prev_Ins <- performance(pred_Prev_Ins, 'sens', 'spec')
plot(auc_plot_Prev_Ins, main=paste("AUC:", auc_val_Prev_Ins@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- bind_rows(result, tibble(Method = " Prev_Ins_model ",
                                   Accuracy = confusionMatrix(data = factor(Prev_Ins_model), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_Prev_Ins@y.values[[1]]))

result


# Since having suffered a damage previously is likely to have higher affirmative response rate, lets create a model for this and check its parameters.

Damage_model <- ifelse(test_set$damage == 1, 1, 0)

## Calculating the accuracy for the Damage Model

confusionMatrix(data = factor(Damage_model), reference = factor(test_set$Response))


## Calculating the AUC for the Damage Model

pred_Damage <- prediction(
  as.numeric(as.character(Damage_model)),as.numeric(as.character(test_set$Response)))

auc_val_Damage <- performance(pred_Damage, "auc")
auc_plot_Damage <- performance(pred_Damage, 'sens', 'spec')
plot(auc_plot_Damage, main=paste("AUC:", auc_val_Damage@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- bind_rows(result, tibble(Method = " Damage_model ",
                                   Accuracy = confusionMatrix(data = factor(Damage_model), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_Damage@y.values[[1]]))

result

# Let us combine these two factors together to see their impact on affirmative response rate, lets create a model for this and check its parameters.

Damage_Insurance_combine <- ifelse(test_set$damage == 1 & test_set$Previously_Insured == 0, 1, 0)

## Calculating the accuracy for the combined Damage and Insurance Model

confusionMatrix(data = factor(Damage_Insurance_combine), reference = factor(test_set$Response))


## Calculating the AUC for the Damage Model

pred_Damage_Insurance_combine <- prediction(
  as.numeric(as.character(Damage_Insurance_combine)),as.numeric(as.character(test_set$Response)))

auc_val_Damage_Insurance_combine <- performance(pred_Damage_Insurance_combine, "auc")
auc_plot_Damage_Insurance_combine <- performance(pred_Damage_Insurance_combine, 'sens', 'spec')
plot(auc_plot_Damage_Insurance_combine, main=paste("AUC:", auc_val_Damage_Insurance_combine@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- bind_rows(result, tibble(Method = " Damage_Insurance_combine ",
                                   Accuracy = confusionMatrix(data = factor(Damage_Insurance_combine), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_Damage_Insurance_combine@y.values[[1]]))

result

# We are using the logistic regression model from Caret package to predict response based on two features - previously insured and damaged.

set.seed(3, sample.kind = "Rounding")

train_set$Response <- as.factor(train_set$Response)

train_glm <- train(Response ~ Previously_Insured + damage, method = "glm", data = train_set)

glm_prediction <- predict(train_glm, test_set)

## Calculating the accuracy for the Previous Insurance Model

confusionMatrix(data = factor(glm_prediction), reference = factor(test_set$Response))

## Calculating the AUC for the Previous Insurance Model

pred_glm <- prediction(
  as.numeric(as.character(glm_prediction)),as.numeric(as.character(test_set$Response)))

auc_val_glm <- performance(pred_glm, "auc")
auc_plot_glm <- performance(pred_glm, 'sens', 'spec')
plot(auc_plot_glm, main=paste("AUC:", auc_val_glm@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- bind_rows(result, tibble(Method = " prediction_glm ",
                                   Accuracy = confusionMatrix(data = factor(glm_prediction), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_glm@y.values[[1]]))

result

# We are using the logistic regression model from Caret package to predict response based on all factors to see if the evaluation parameters improve.

set.seed(3, sample.kind = "Rounding")

train_set$Response <- as.factor(train_set$Response)

train_glm_all <- train(Response ~ ., method = "glm", data = train_set)

glm_prediction_all <- predict(train_glm_all, test_set)

## Calculating the accuracy for the Previous Insurance Model

confusionMatrix(data = factor(glm_prediction_all), reference = factor(test_set$Response))

## Calculating the AUC for the Previous Insurance Model

pred_glm_all <- prediction(
  as.numeric(as.character(glm_prediction_all)),as.numeric(as.character(test_set$Response)))

auc_val_glm_all <- performance(pred_glm_all, "auc")
auc_plot_glm_all <- performance(pred_glm_all, 'sens', 'spec')
plot(auc_plot_glm_all, main=paste("AUC:", auc_val_glm_all@y.values[[1]]))

## Creating the results table with evaluation metrics

result <- bind_rows(result, tibble(Method = " prediction_glm_all ",
                                   Accuracy = confusionMatrix(data = factor(glm_prediction_all), reference = factor(test_set$Response)) $overall["Accuracy"], AUC = auc_val_glm_all@y.values[[1]]))

result

# We are using the XGBoost Package which has already been installed from CRAN library.

set.seed(1234, sample.kind = "Rounding")

library(xgboost)

# We convert our train and test set in XGBoost compliant dataset format.

xgb_train <- xgb.DMatrix(
  as.matrix(train_set[, colnames(train_set) != "Response"]), 
  label = as.numeric(as.character(train_set$Response)))

xgb_test <- xgb.DMatrix(
  as.matrix(test_set[, colnames(test_set) != "Response"]), 
  label = as.numeric(as.character(test_set$Response)))


# We create the XGBoost model by specifying the tuning parameters. We used the error (ie Binary classification error rate which measures #wrong cases/#all cases) minimization as our evaluation metric.

xgb_params <- list(
  objective = "binary:logistic", 
  eta = 1, 
  max.depth = 2, 
  nthread = 5, 
  eval_metric = "error")

xgb_model <- xgb.train(
  data = xgb_train, 
  params = xgb_params, 
  watchlist = list(test = xgb_test), 
  nrounds = 200, 
  early_stopping_rounds = 50,
  verbosity = 0,
  print_every_n = 100,
  silent = T)

# XGBoost model suggests the relative importance of various features and confirms that Previous Insurance and Damage are the two most pertinent predictors.

feature_imp_xgb <- xgb.importance(colnames(train_set), model = xgb_model)


xgb.plot.importance(feature_imp_xgb, rel_to_first = TRUE, xlab = "Relative importance")

predictions_xgb = predict(
  xgb_model, 
  newdata = as.matrix(test_set[, colnames(test_set) != "Response"]), 
  ntreelimit = xgb_model$bestInd)


## Calculating the AUC for the XGBoost Model

pred_xgb <- prediction(
  as.numeric(as.character(predictions_xgb)),as.numeric(as.character(test_set$Response)))


auc_val_xgb <- performance(pred_xgb, "auc")
auc_plot_xgb <- performance(pred_xgb, 'sens', 'spec')
plot(auc_plot_xgb, main=paste("AUC:", auc_val_xgb@y.values[[1]]))

## Creating the results table with evaluation metrics. Accuracy is the 1-error for selected model which stopped at 25th iteration.

result <- bind_rows(result, tibble(Method = " predictions_xgb ",
                                   Accuracy = 0.877201, AUC = auc_val_xgb@y.values[[1]]))

result
