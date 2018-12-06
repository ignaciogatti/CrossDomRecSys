import pandas as pd
import numpy as np
from space_model.EmbeddingVectorSpace import EmbeddingVectorTargetSpace


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres_description.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating_description.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')

embedding_space = EmbeddingVectorTargetSpace(df_item_origin=df_ml_movies, df_item_target=df_bx_book, df_ratings=df_movie_ratings)

movie_embedding_space = embedding_space.origin_embedding_space()
print(movie_embedding_space.shape)
np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/GLoVe/movie_embedding_space', movie_embedding_space)

book_embedding_space = embedding_space.target_embedding_space()
print(book_embedding_space.shape)
np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/GLoVe/book_embedding_space', book_embedding_space)

users_embedding_space = embedding_space.build_users_profiles()
users_profile_matrix = np.vstack(users_embedding_space.values())
print(users_profile_matrix.shape)

np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/GLoVe/user_embedding_space', users_profile_matrix)

