library(tm)
library(SnowballC)
library(caret)



#load the data set
dataset <- read.csv("MachineLearningChallengeData.csv")
dataset$commentlength <- nchar(gsub("\\s", "", dataset$comment))
#remove unnecessary columns
dataset <- dataset[, c(2, 3, 5)]

#change columns names
names(dataset) <- c("comment", "toxic", "commentLength")

#distribution of toxic and non-toxic comments
print(table(dataset$toxic))

# tempray data for testing
data <- dataset



# Remove URLs starting with "https://"
data$comment <- gsub("https?://[^\\s]+", "", data$comment)

# Remove URLs starting with "http://"
data$comment <- gsub("http?://[^\\s]+", "", data$comment)

# Remove punctuation marks
data$comment <- gsub("[[:punct:]]", "", data$comment)

# Remove extra white spaces (reducing multiple consecutive spaces to a single space)
data$comment <- gsub("\\s{2,}", " ", data$comment)

# Convert all characters in the 'comment' column to lowercase
data$comment <- tolower(data$comment) 

# Remove names of days
data$comment <- gsub("\\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\\b", "", data$comment, ignore.case = TRUE)

# Remove names of months
data$comment <- gsub("\\b(january|february|march|april|may|june|july|august|september|october|november|december)\\b", "", data$comment, ignore.case = TRUE)





# Remove English stopwords
data$comment <- removeWords(data$comment, stopwords("en"))

# Remove extra whitespaces (reducing multiple consecutive spaces to a single space)
data$comment <- gsub("\\s{2,}", " ", data$comment)

# Remove digits (numeric characters)
data$comment <- gsub("[0-9]", "", data$comment)

#remove words with only one letter
data$comment <- gsub("\\b\\w{1}\\b", "", data$comment)




#install.packages("tokenizers")
#install.packages("SnowballC")
library(SnowballC)
library(tokenizers)

# Tokenize the text column using the tokenizers package
tokenized_text <- tokenize_words(data$comment)

# Stem each token using the SnowballC package
stemmed_tokens <- lapply(tokenized_text, function(tokens) {
  wordStem(tokens, language = "en")  # "en" indicates English language
})

# Combine stemmed tokens back into text
stemmed_text <- sapply(stemmed_tokens, paste, collapse = " ")

# Replace the original text column with the stemmed text
data$comment <- stemmed_text









# Create a document-term matrix (DTM) using TF-IDF
tfidf_vectorizer <- function(data) {
  corpus <- Corpus(VectorSource(data))
  dtm <- DocumentTermMatrix(corpus)
  dtm_tfidf <- weightTfIdf(dtm)
  
  # Reduce the size of the TF-IDF matrix by keeping only the top 1000 features
  dtm_tfidf_reduced <- removeSparseTerms(dtm_tfidf, sparse = 0.999)
  
  return(as.matrix(dtm_tfidf_reduced))
}

# Apply TF-IDF vectorizer on data
tfidf_data <- tfidf_vectorizer(data$comment)


# Define labels
target <- data$toxic


# Combine label and TF-IDF matrix into a data frame
combined_data <- data.frame(label = target, tfidf_data)


# Split the data into training and testing sets
set.seed(42)  # for reproducibility
train_index <- createDataPartition(combined_data$label, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]

# Train logistic regression model
model <- glm(label ~ ., data = train_data, family = binomial(link = "logit"))

# Make predictions
predictions <- ifelse(predict(model, newdata = test_data, type = "response") > 0.5, 1, 0)

# Print list of predictions
print(predictions)


# Evaluate model performance
confusion_matrix <- confusionMatrix(factor(predictions, levels = c(0, 1)), factor(test_data$label, levels = c(0, 1)))

# Print confusion matrix
print(confusion_matrix)


# Extract precision, recall, and other metrics
precision <- confusion_matrix$byClass["Pos Pred Value"]
recall <- confusion_matrix$byClass["Sensitivity"]
f1_score <- confusion_matrix$byClass["F1"]
accuracy <- confusion_matrix$overall["Accuracy"]

# Additional metrics
specificity <- confusion_matrix$byClass["Specificity"]
balanced_accuracy <- confusion_matrix$byClass["Balanced Accuracy"]
kappa <- confusion_matrix$overall["Kappa"]

# Print metrics
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
cat("Accuracy:", accuracy, "\n")
cat("Specificity:", specificity, "\n")
cat("Balanced Accuracy:", balanced_accuracy, "\n")
cat("Kappa:", kappa, "\n")



#Train Naive Bayes model
library(e1071)
naive_bayes_model <- naiveBayes(as.factor(label) ~ ., data = train_data)

# Make predictions
naive_bayes_predictions <- predict(naive_bayes_model, newdata = test_data)

# Calculate confusion matrix
naive_bayes_confusion_matrix <- confusionMatrix(naive_bayes_predictions, factor(test_data$label, levels = c(0, 1)))

# Print confusion matrix
print(naive_bayes_confusion_matrix)

# Extract metrics for Naive Bayes model
naive_bayes_precision <- naive_bayes_confusion_matrix$byClass["Pos Pred Value"]
naive_bayes_recall <- naive_bayes_confusion_matrix$byClass["Sensitivity"]
naive_bayes_f1_score <- naive_bayes_confusion_matrix$byClass["F1"]
naive_bayes_accuracy <- naive_bayes_confusion_matrix$overall["Accuracy"]
naive_bayes_specificity <- naive_bayes_confusion_matrix$byClass["Specificity"]
naive_bayes_balanced_accuracy <- naive_bayes_confusion_matrix$byClass["Balanced Accuracy"]
naive_bayes_kappa <- naive_bayes_confusion_matrix$overall["Kappa"]

# Print metrics for Naive Bayes model
cat("Naive Bayes Metrics:\n")
cat("Precision:", naive_bayes_precision, "\n")
cat("Recall:", naive_bayes_recall, "\n")
cat("F1 Score:", naive_bayes_f1_score, "\n")
cat("Accuracy:", naive_bayes_accuracy, "\n")
cat("Specificity:", naive_bayes_specificity, "\n")
cat("Balanced Accuracy:", naive_bayes_balanced_accuracy, "\n")
cat("Kappa:", naive_bayes_kappa, "\n")





# Train SVM model
svm_model <- svm(as.factor(label) ~ ., data = train_data)

# Make predictions
svm_predictions <- predict(svm_model, newdata = test_data)

# Calculate confusion matrix
svm_confusion_matrix <- confusionMatrix(svm_predictions, factor(test_data$label, levels = c(0, 1)))

# Print confusion matrix
print(svm_confusion_matrix)









































