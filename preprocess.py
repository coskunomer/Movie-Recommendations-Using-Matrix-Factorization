'''
!!!!!!
Because of the GitHub's file size limitages, the datasets are not on this directory.
You can go and download the data from https://grouplens.org/datasets/movielens/25m/
After downloading, just put them into datasets/ folder.
'''

import pandas as pd
import numpy as np
import pickle as pkl
from collections import Counter
from sklearn.model_selection import train_test_split
import time

ratings = pd.read_csv("datasets/ratings.csv")
ratings.drop(columns=["timestamp"], inplace=True)
ratings["userId"] = np.array(ratings.userId.tolist()) - 1

correctedMovieId = {}
for idx, movieId in enumerate(list(set(ratings.movieId.tolist()))):
    correctedMovieId[movieId] = idx

newIds = []
for value in ratings["movieId"].tolist():
    newIds.append(correctedMovieId[value])

ratings["movieId"] = newIds

n = 10000
m = 2000

users = [userId for userId, count in Counter(ratings.userId).most_common(n)]
movies = [movieId for movieId, count in Counter(ratings.movieId).most_common(m)]

ratings_small = ratings[ratings.movieId.isin(movies) & ratings.userId.isin(users)].copy()

correctedMovieId = {}
correctedUserId = {}

for idx, movieId in enumerate(list(set(ratings_small.movieId.tolist()))):
    correctedMovieId[movieId] = idx
for idx, userId in enumerate(list(set(ratings_small.userId.tolist()))):
    correctedUserId[userId] = idx

newIds = []
for value in ratings_small["movieId"].tolist():
    newIds.append(correctedMovieId[value])
ratings_small["movieId"] = newIds

newIds = []
for value in ratings_small["userId"].tolist():
    newIds.append(correctedUserId[value])

ratings_small["userId"] = newIds
ratings_small = ratings_small.reset_index()

ratings_train, ratings_test = train_test_split(ratings, test_size=0.2)
ratings_small_train, ratings_small_test = train_test_split(ratings_small, test_size=0.2)

user2movie = {}
movie2user = {}
usermovie2rating_train = {}
usermovie2rating_test = {}

ratings_train = ratings_train.reset_index()
ratings_test = ratings_test.reset_index()
ratings_small_train = ratings_small_train.reset_index()
ratings_small_test = ratings_small_test.reset_index()

start = time.time()
for idx, row in ratings_train.iterrows():
    if idx % 100000 == 0:
        print(idx)
    if time.time() - start % 1000 == 0:
        time.sleep(60)
    user2movie[int(row["userId"])] = user2movie.get(int(row["userId"]), []) + [int(row["movieId"])]
    movie2user[int(row["movieId"])] = movie2user.get(int(row["movieId"]), []) + [int(row["userId"])]
    usermovie2rating_train[(int(row["userId"]), int(row["movieId"]))] = row["rating"]

for idx, row in ratings_test.iterrows():
    if idx % 100000 == 0:
        print(idx)
    if time.time() - start % 1000 == 0:
        time.sleep(60)
    usermovie2rating_test[(int(row["userId"]), int(row["movieId"]))] = row["rating"]

user2movie_small = {}
movie2user_small = {}
usermovie2rating_small_train = {}
usermovie2rating_small_test = {}
for idx, row in ratings_small_train.iterrows():
    if idx % 100000 == 0:
        print(idx)
    if time.time() - start % 1000 == 0:
        time.sleep(60)
    user2movie_small[int(row["userId"])] = user2movie_small.get(int(row["userId"]), []) + [int(row["movieId"])]
    movie2user_small[int(row["movieId"])] = movie2user_small.get(int(row["movieId"]), []) + [int(row["userId"])]
    usermovie2rating_small_train[(int(row["userId"]), int(row["movieId"]))] = row["rating"]
for idx, row in ratings_small_test.iterrows():
    if idx % 100000 == 0:
        print(idx)
    if time.time() - start % 1000 == 0:
        time.sleep(60)
    usermovie2rating_small_test[(int(row["userId"]), int(row["movieId"]))] = row["rating"]

with open("preprocessed_dataset/user2movie.pkl", "wb") as js:
    pkl.dump(user2movie, js)
with open("preprocessed_dataset/movie2user.pkl", "wb") as js:
    pkl.dump(movie2user, js)
with open("preprocessed_dataset/usermovie2rating_train.pkl", "wb") as js:
    pkl.dump(usermovie2rating_train, js)
with open("preprocessed_dataset/usermovie2rating_test.pkl", "wb") as js:
    pkl.dump(usermovie2rating_test, js)
with open("preprocessed_dataset/user2movie_small.pkl", "wb") as js:
    pkl.dump(user2movie_small, js)
with open("preprocessed_dataset/movie2user_small.pkl", "wb") as js:
    pkl.dump(movie2user_small, js)
with open("preprocessed_dataset/usermovie2rating_small_train.pkl", "wb") as js:
    pkl.dump(usermovie2rating_small_train, js)
with open("preprocessed_dataset/usermovie2rating_small_test.pkl", "wb") as js:
    pkl.dump(usermovie2rating_small_test, js)
ratings_train.to_csv("preprocessed_dataset/ratings_train.csv")
ratings_test.to_csv("preprocessed_dataset/ratings_train_test.csv")
ratings_small_train.to_csv("preprocessed_dataset/ratings_small_train.csv")
ratings_small_test.to_csv("preprocessed_dataset/ratings_small_test.csv")