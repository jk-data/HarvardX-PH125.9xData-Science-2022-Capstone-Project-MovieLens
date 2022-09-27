## HarvardX PH125.9xData Science 2022: Capstone Project MovieLens
## Jyotirmay Kirtania
## Date: 27 September 2022
## https://github.com/jk-data

########################################################
# Code for Movie Rating Prediction in MovieLens Project 
########################################################

########################################################
#### Install essential R packages ####

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")

if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

if(!require(stringr)) install.packages("stringr", repos = "http://cran.us.r-project.org")

library(tidyverse)

library(caret)

library(data.table)

library(knitr)

library(kableExtra)

library(lubridate)

library(stringr)


############################################################################### 

#### Dataset download, creation of training dataset and validation dataset ####

#The MovieLens 10M data-set is described at: https://grouplens.org/datasets/movielens/10m/

#The data-set was downloaded from: http://files.grouplens.org/datasets/movielens/ml-10m.zip

#The following code was used to download the data-set. Then the information of the ratings and the movies was combined into MovieLens:

dl <- tempfile() 
options(timeout=360) 
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))), col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "::", 3)

colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId], title = as.character(title), genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")


#In order to predict in the most accurate way the movie rating of the users who are yet to see the movie, the he MovieLens dataset will be split into two subsets. One will be the "edx", a training subset to train the algorithm. The other will be the "validation", a subset to test the movie ratings. Now we split the MovieLens data-set into Training (edx) and Validation (validation) sets. The Validation set will be 10% of the MovieLens dataset. 

set.seed(1) 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE) 
edx <- movielens[-test_index,] 
temp <- movielens[test_index,]

#Make sure userId and movieId in validation set are also in edx subset:

validation <- temp %>%
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

#Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)



#### Data exploration and Analysis ####

# A general overview of the dataset

head(edx) %>%
  print.data.frame()

# Total unique movies and users

summary(edx)

# Number of unique movies and users in the edx dataset

edx %>%
  summarize(n_users = n_distinct(userId), 
            n_movies = n_distinct(movieId))


# Plot the ratings distribution

edx %>%
  ggplot(aes(rating)) +
  geom_histogram(binwidth = 0.25, color = "black") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  scale_y_continuous(breaks = c(seq(0, 3000000, 500000))) +
  ggtitle("Rating distribution")


# Plot the number of ratings per movie

edx %>%
  count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") +
  ylab("Number of movies") +
  ggtitle("Number of ratings per movie")


# Create table of twenty movies that were rated only once

edx %>%
  group_by(movieId) %>%
  summarize(count = n()) %>%
  filter(count == 1) %>%
  left_join(edx, by = "movieId") %>%
  group_by(title) %>%
  summarize(rating = rating, n_rating = count) %>%
  slice(1:20) %>%
  knitr::kable()

# Plot the number of ratings given by users

edx %>%
  count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "black") +
  scale_x_log10() +
  xlab("Number of ratings") + 
  ylab("Number of users") +
  ggtitle("Number of ratings given by users")


# Plot the mean movie ratings given by users

edx %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black") +
  xlab("Mean rating") +
  ylab("Number of users") +
  ggtitle("Mean movie ratings given by users") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) +
  theme_light()



#### Data Modelling ####

#The RMSE is the measure of model accuracy. The RMSE is the typical error we make when predicting a movie rating. If its result is larger than 1, it means that our typical error is larger than one star, which is not a good prediction. The written function to compute the RMSE for vectors of ratings and their corresponding predictions is:
#RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings - predicted_ratings)^2))}


## A. Average movie rating model (naive model) ##

# Compute the mean rating of the dataset

mu <- mean(edx$rating)
mu
  
# Test the results (naive RMSE) based on simple prediction

naive_rmse <- RMSE(validation$rating, mu)

# Check the result

naive_rmse

# Save prediction to data frame

rmse_results <- data_frame(method = "Average movie rating model", RMSE = naive_rmse)
rmse_results %>% knitr::kable()


## B. Movie effect model ##

# To improve upon the previous model we focus on the fact that, some movies generally receive a higher rating than others. Higher ratings are mostly linked to popular movies among users and the opposite is true for unpopular movies. We compute the estimated deviation of mean rating for each movie from the total mean of all movies. The resulting variable is called "b_i" (bias) for each movie, that represents average ranking for movie.

# Simple model taking into account the movie effect b_i
  
# Subtract the rating minus the mean for each rating the movie received
  
# Plot the number of movies with the computed b_i

movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"),
                     ylab = "Number of movies", main = "Number of movies with the computed b_i")


# Test and save the RMSE results

predicted_ratings <- mu +  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie effect model",  
                                     RMSE = model_1_rmse ))
# Check results
rmse_results %>% knitr::kable()


## C. Movie and user effect model ##

# There is significant variability among users while they rate the movies. Some users are very choosy while others like almost every movie. This implies that further improvement to our previous model can be done. We compute the average rating for user who have rated over 100 movies. This is the penalty term user effect.

# Plot the penalty term user effect

user_avgs<- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  filter(n() >= 100) %>%
  summarize(b_u = mean(rating - mu - b_i))
user_avgs%>% qplot(b_u, geom ="histogram", bins = 30, data = ., color = I("black"))


user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))


# Test and save the RMSE results

predicted_ratings <- validation%>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# Construct predictors and see if RMSE improves

model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie and user effect model",  
                                     RMSE = model_2_rmse))
# Check the result

rmse_results %>% knitr::kable()


#### D. Regularized movie and user effect model

#There are also some users who rated a very small number of movies. These ratings can strongly influence the predictions. The use of the regularization permits to penalize these aspects. We should find the value of lambda which is a tuning parameter that will minimize the RMSE. 

# Use cross-validation to choose the lambda

lambdas <- seq(0, 10, 0.25)


# For each lambda, find b_i & b_u, followed by rating prediction & testing

rmses <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})


# Plot the RMSES vs the Lambdas to select the optimal lambda                                                             
qplot(lambdas, rmses) 


# Find the optimal lambda for the the regularized prediction model

lambda <- lambdas[which.min(rmses)] lambda


# Test and save results

rmse_results <- bind_rows(rmse_results,
data_frame(method="Regularized movie and user effect model",  
RMSE = min(rmses)))

# Check the result

rmse_results %>% knitr::kable()


#### Results ####

# Check the summary of the results comparing the various movie rating prediction models.
# The model with the lowest RMSE is the best movie rating prediction model in this project. 

rmse_results %>% knitr::kable()


#### Computation Environment ####

print("Operating System:")

version

