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


set.seed(42)  # for reproducibility
train_index <- createDataPartition(data$toxic, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# Create a document-term matrix (DTM) using TF-IDF
tfidf_vectorizer <- function(data) {
  corpus <- Corpus(VectorSource(data))
  dtm <- DocumentTermMatrix(corpus)
  dtm_tfidf <- weightTfIdf(dtm)
  return(as.matrix(dtm_tfidf))
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























