#############################################################
# Create edx set, validation set, and submission file
#############################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#------------------------- START OF PROJECT---------------------------


#Define RMSE function used to calculate testing of algorithm
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


#----- 1st Algorithm Just average prediction ---

#Calculate average rating accorss all rating
mu_hat <- mean(edx$rating)

#Calculate naive RMSE if mu_hat is used to predict all rating on validation set
naive_rmse <- RMSE(validation$rating, mu_hat)

#Create RMSE results table to store all RMSE for different algorithm
rmse_results <- data_frame(method = "1st: Just the average", RMSE = naive_rmse)

#Display RMSE for Naive RMSE alsorithm (using average to predict movie rating)
options(pillar.sigfig = 6)
pillar::pillar(rmse_results)

#------ 2nd Algorithm Movie Effect prediction ---------------------------

#Calculate the Movie effect which influence the rating
movie_avgs <- edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))

#Using Movie effect to calculate rating prediction on validation set
predicted_ratings <- mu_hat + validation%>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

#Calculate RMSE result and store in RMSE result table
model_1_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="2nd: Movie Effect Model",
                                     RMSE = model_1_rmse ))

#Display RMSE result for 1st and 2nd algorithm
#Notice significant better result in 2nd Algoritm
options(pillar.sigfig = 6)
pillar::pillar(rmse_results)

#------- 3rd Algorithm Movie Effect + User Effect prediction -----


#Calculate the user effect which affect the final rating:
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

#Using Movie effect and User effect to predict the rating in validation set
predicted_ratings <-mu_hat + validation %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred =  b_i + b_u) %>%
  .$pred

#Calculate RMSE result and store in RMSE result table
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="3rd: Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))


#Display RMSE result for 1st, 2nd and 3rd algorithm
#Notice significant better result in 3rd Algoritm
options(pillar.sigfig = 6)
pillar::pillar(rmse_results)


#-------4th Algorithm Movie, User and Genres Effect prediction-------------------------------


#Calculate the genres effect which influence the rating
genres_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs,by = 'userId') %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i-b_u))

#Using Movie effect,User effect and Genres to predict the rating in validation set
temp <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId')

temp2<-left_join(temp,genres_avgs,by = 'genres')
predicted_ratings <- mu_hat +temp2 %>% mutate(pred = b_i+b_u+b_g) %>% .$pred


#Calculate RMSE result and store in RMSE result table
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="4th: Movie + User Effects Model + Genres",  
                                     RMSE = model_2_rmse ))

#Display RMSE result for 1st, 2nd, 3rd  and 4th algorithm
#Notice significant better result in 4th Algoritm
options(pillar.sigfig = 6)
pillar::pillar(rmse_results)


#Conclusion: From above 4 algorithm, the algorithm that perform best
#is the 4th one, which account for Movie + User + Genres Effect
#with RMSE of 0.864
