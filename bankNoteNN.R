# ====================================================
# Bank Note Fraud Detection using Machine Learning
# ====================================================
# This program demonstrates how to build and compare two machine learning models
# (Neural Network and Random Forest) to classify bank notes as genuine or fraudulent.
# The dataset used is the UCI Bank Authentication Dataset.
# ====================================================

# --- Load Required Libraries ---
library(data.table)    # Fast data reading and manipulation
library(ggplot2)       # Data visualization
library(neuralnet)     # Neural network modeling
library(randomForest)  # Random forest modeling
library(caTools)       # Data splitting
library(caret)         # Model evaluation and metrics
library(knitr)         # Professional table formatting
library(dplyr)         # Data manipulation
library(reshape2)      # Data reshaping for visualization

# --- 1️⃣ Exploratory Data Analysis (EDA) ---
# Goal: Understand the dataset, check for missing values, and visualize distributions.

# Load the dataset
data <- fread("bank_note_data.csv")

# Display basic information about the dataset
summary(data)  # Summary statistics for each column
str(data)      # Structure of the dataset

# Check for missing values
na_counts <- sapply(data, function(x) sum(is.na(x)))
print(na_counts)  # Print count of missing values per column

# Handle missing values (if any) by imputing with the mean
for (col in names(data)) {
  if (is.numeric(data[[col]])) {
    data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
  }
}

# Visualize the distribution of the target variable (Class)
pl1 <- ggplot(data, aes(x = factor(Class))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Class Distribution",
       x = "Class (0 = Genuine, 1 = Fraudulent)",
       y = "Count")
print(pl1)

# Identify feature names (all columns except 'Class')
feature_names <- setdiff(names(data), "Class")

# Visualize feature distributions using histograms
for (feature in feature_names) {
  p <- ggplot(data, aes_string(x = feature)) +
    geom_histogram(bins = 30, fill = "skyblue", color = "black") +
    labs(title = paste("Histogram of", feature))
  print(p)
}

# Visualize feature distributions using boxplots
for (feature in feature_names) {
  p <- ggplot(data, aes_string(y = feature)) +
    geom_boxplot(fill = "tomato", color = "black") +
    labs(title = paste("Boxplot of", feature))
  print(p)
}

# Analyze feature correlations
cor_matrix <- cor(data[, ..feature_names])  # Compute correlation matrix
print(cor_matrix)  # Print correlation matrix

# Visualize the correlation matrix
melted_cor <- melt(cor_matrix)  # Reshape for visualization
pl2 <- ggplot(melted_cor, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Matrix") +
  theme_minimal()
print(pl2)

# --- 2️⃣ Data Preprocessing ---
# Goal: Prepare the data for modeling by standardizing features and splitting into train/test sets.

# Standardize numeric features (mean = 0, standard deviation = 1)
data_std <- copy(data)
data_std[, (feature_names) := lapply(.SD, scale), .SDcols = feature_names]

# Split the dataset into training (80%) and testing (20%) sets
set.seed(123)  # For reproducibility
data_std$Class <- as.factor(data_std$Class)  # Convert target to factor
split <- sample.split(data_std$Class, SplitRatio = 0.8)
train_data <- subset(data_std, split == TRUE)  # Training set
test_data  <- subset(data_std, split == FALSE)  # Testing set

# --- 3️⃣ Neural Network Model ---
# Goal: Build and tune a neural network model to classify bank notes.

# Convert target variable to numeric for neural network compatibility
train_data$Class <- as.numeric(as.character(train_data$Class))

# Define the neural network formula
nn_formula <- reformulate(feature_names, "Class")

# Tune hidden layer configurations
hidden_configs <- list("3" = c(3), "5" = c(5), "3_2" = c(3, 2))  # Different architectures
nn_tuning_results <- list()  # Store results for each configuration

cat("Neural Network Hyperparameter Tuning:\n")
for (name in names(hidden_configs)) {
  cat("Training NN with hidden layers:", hidden_configs[[name]], "\n")
  
  # Train the neural network
  nn_model <- neuralnet(
    nn_formula,
    data = train_data,
    hidden = hidden_configs[[name]],
    linear.output = FALSE,
    stepmax = 1e6
  )
  
  # Evaluate the model on the test set
  nn_pred <- neuralnet::compute(nn_model, test_data[, feature_names, with = FALSE])
  predicted <- nn_pred$net.result
  actual <- as.numeric(as.character(test_data$Class))
  
  # Calculate Mean Squared Error (MSE)
  mse_val <- mean((predicted - actual)^2)
  nn_tuning_results[[name]] <- list(model = nn_model, mse = mse_val)
  cat("   MSE:", round(mse_val, 4), "\n")
}

# Select the best model based on the lowest MSE
best_config <- names(which.min(sapply(nn_tuning_results, \(x) x$mse)))
best_nn_model <- nn_tuning_results[[best_config]]$model
cat("Best Neural Network Configuration:", best_config, "\n")

# Visualize the best neural network architecture
plot(best_nn_model)

# --- 4️⃣ Neural Network Evaluation ---
# Goal: Evaluate the performance of the best neural network model.

# Generate predictions on the test set
nn_test_pred <- neuralnet::compute(best_nn_model, test_data[, feature_names, with = FALSE])$net.result
nn_test_class <- ifelse(nn_test_pred > 0.5, 1, 0)  # Convert probabilities to classes

# Create a confusion matrix
conf_matrix_nn <- table(Predicted = nn_test_class, Actual = test_data$Class)
conf_nn <- confusionMatrix(factor(nn_test_class, levels = 0:1),
                           factor(test_data$Class, levels = 0:1))
print(conf_nn)

# Extract performance metrics
nn_metrics <- data.frame(
  Model = "Neural Network",
  Accuracy = round(conf_nn$overall['Accuracy'], 4),
  Precision = round(conf_nn$byClass['Pos Pred Value'], 4),
  Recall = round(conf_nn$byClass['Sensitivity'], 4),
  F1_Score = round(conf_nn$byClass['F1'], 4)
)

# Display metrics in a neat table
library(kableExtra)
nn_metrics %>%
  kable(caption = "Neural Network Performance Metrics") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))

# --- 5️⃣ Random Forest Model ---
# Goal: Build and evaluate a random forest model for comparison.

# Train the random forest model
set.seed(456)  # For reproducibility
rf_model <- randomForest(Class ~ ., 
                         data = train_data %>% mutate(Class = factor(Class, levels = 0:1)),
                         ntree = 100)

# Generate predictions on the test set
rf_pred <- predict(rf_model, test_data)
rf_cm <- confusionMatrix(rf_pred, test_data$Class)
print(rf_cm)

# Extract performance metrics
rf_metrics <- data.frame(
  Model = "Random Forest",
  Accuracy = round(rf_cm$overall['Accuracy'], 4),
  Precision = round(rf_cm$byClass['Pos Pred Value'], 4),
  Recall = round(rf_cm$byClass['Sensitivity'], 4),
  F1_Score = round(rf_cm$byClass['F1'], 4)
)

# Compare model performance
comparison <- rbind(nn_metrics, rf_metrics)
comparison %>%
  kable(caption = "Model Performance Comparison") %>%
  kable_styling(bootstrap_options = c("striped", "hover"))
kable(comparison)
# --- Bonus Features ---

# Bonus 1: Visualize Neural Network Training Error
if (exists("best_nn_model") && !is.null(best_nn_model$result.matrix)) {
  error_df <- data.frame(
    Step = seq_along(best_nn_model$result.matrix["error", ]),
    Error = best_nn_model$result.matrix["error", ]
  )
  
  ggplot(error_df, aes(x = Step, y = Error)) +
    geom_line(color = "navy") +
    labs(title = "Neural Network Training Error Convergence",
         x = "Training Step", y = "Error") +
    theme_minimal()
}

# Bonus 2: 5-Fold Cross-Validation for Neural Network
best_hidden <- hidden_configs[[best_config]]  # Use best configuration
folds <- createFolds(data_std$Class, k = 5)
cv_accuracies <- sapply(folds, function(fold) {
  cv_train <- data_std[-fold, ]
  cv_test <- data_std[fold, ]
  
  # Train and evaluate the model
  cv_model <- neuralnet(
    nn_formula,
    data = cv_train,
    hidden = best_hidden,
    linear.output = FALSE,
    stepmax = 1e6
  )
  
  predictions <- neuralnet::compute(cv_model, cv_test[, feature_names, with = FALSE])$net.result
  predicted_class <- ifelse(predictions > 0.5, 1, 0)
  actual_class <- as.numeric(as.character(cv_test$Class))
  
  mean(predicted_class == actual_class)
})

cat("\n5-Fold Cross-Validation Results:\n",
    "Mean Accuracy:", round(mean(cv_accuracies), 4), "\n",
    "Standard Deviation:", round(sd(cv_accuracies), 4))

# Bonus 3: Visualize Random Forest Feature Importance
varImpPlot(rf_model,
           main = "Random Forest Feature Importance",
           col = "steelblue",
           pch = 16)