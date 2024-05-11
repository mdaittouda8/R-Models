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







# Split the data into training and testing sets
set.seed(42)  # for reproducibility
train_index <- createDataPartition(data$toxic, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Create a document-term matrix (DTM) using TF-IDF
tfidf_vectorizer <- function(data) {
  corpus <- Corpus(VectorSource(data))
  dtm <- DocumentTermMatrix(corpus)
  dtm_tfidf <- weightTfIdf(dtm)
  
  # Reduce the size of the TF-IDF matrix by keeping only the top 1000 features
  dtm_tfidf_reduced <- removeSparseTerms(dtm_tfidf, sparse = 0.999)
  
  return(as.matrix(dtm_tfidf_reduced))
}

# Apply TF-IDF vectorizer on training and testing data
X_train <- tfidf_vectorizer(train_data$comment)
X_test <- tfidf_vectorizer(test_data$comment)

# Define labels
y_train <- train_data$toxic
y_test <- test_data$toxic

# Combine label and TF-IDF matrix into a data frame
train_df <- data.frame(label = y_train, X_train)
test_df <- data.frame(label = y_test, X_test)

# Train logistic regression model
model <- glm(label ~ ., data = train_df, family = binomial(link = "logit"))

# Make predictions
predictions <- ifelse(predict(model, newdata = test_df, type = "response") > 0.5, 1, 0)

# Evaluate model performance
confusion_matrix <- confusionMatrix(predictions, y_test)
print(confusion_matrix)






