
  ### Toxic Classifier: 
  
  #### 1- Data Set Loading and Packages :
  
# Load required libraries
library(ggplot2)
library(tm)
library(SnowballC)
library(caret)
library(tokenizers)

#load the data set
dataset <- read.csv("MachineLearningChallengeData.csv")



#### 2- Data Preprocessing and Cleaning:

# Create a new column 'commentlength' with the number of letters (excluding spaces) in each comment
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



#### 3- Feature engineering:
##### Comment Length:

# Calculate average comment length for toxic and non-toxic comments
avg_length_toxic <- mean(data$commentLength[data$toxic == 1])
avg_length_non_toxic <- mean(data$commentLength[data$toxic == 0])

# Print the average lengths
cat("Average comment length for toxic comments:", avg_length_toxic, "\n")
cat("Average comment length for non-toxic comments:", avg_length_non_toxic, "\n")

# Subset data for toxic and non-toxic comments
toxic_comments <- data$commentLength[data$toxic == 1]
non_toxic_comments <- data$commentLength[data$toxic == 0]

# Create histograms 
ggplot() +
  geom_histogram(aes(x = non_toxic_comments), fill = "blue", alpha = 0.5, binwidth = 10) +
  geom_histogram(aes(x = toxic_comments), fill = "red", alpha = 0.5, binwidth = 10) +
  labs(x = "Comment Length", y = "Frequency", title = "Distribution of Comment Lengths for Toxic and Non-Toxic Comments") +
  scale_x_continuous(breaks = seq(0, 1000, by = 100), limits = c(0, 800)) +  # Adjust x-axis limits
  scale_y_continuous(breaks = seq(0, 10000, by = 1000)) +
  theme_minimal()



##### Comment Sentiment:

# Load the syuzhet library
library(syuzhet)

# Get NRC sentiment scores
sentiment <- get_sentiment(data$comment, method = "nrc")

# Convert the sentiment scores to a dataframe
sentiment_scores <- as.data.frame(sentiment)

# combined data to setiment_scores
data <- cbind(data, sentiment_scores)

# explore relationship between sentiment and comment toxicity
# Calculate average comment length for toxic and non-toxic comments
avg_sentiment_toxic <- mean(data$sentiment[data$toxic == 1])
avg_sentiment_non_toxic <- mean(data$sentiment[data$toxic == 0])

# Print the average lengths
cat("Average comment sentiment for toxic comments:", avg_sentiment_toxic, "\n")
cat("Average comment sentiment for non-toxic comments:", avg_sentiment_non_toxic, "\n")



##### TF-IDF Matrix:

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



#### 4-Model Building:


# Split the data into training and testing sets
set.seed(42)  # for reproducibility
train_index <- createDataPartition(combined_data$label, p = 0.8, list = FALSE)
train_data <- combined_data[train_index, ]
test_data <- combined_data[-train_index, ]


##### Naive Bayes:
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


