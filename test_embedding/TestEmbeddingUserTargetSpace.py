import pandas as pd
import numpy as np
from space_model.EmbeddingTFIDFVectorSpace import EmbeddingTFIDFauthorUserTargetSpace
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres_description.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating_description.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')

#Define influence graph
g_social = InfluenceGraph()

embedding_space = EmbeddingTFIDFauthorUserTargetSpace(df_item_origin=df_ml_movies, df_ratings=df_movie_ratings,
                                                  df_item_target=df_bx_book, g_social=g_social.get_influence_graph())

# Origin space
print('Origin space')
movie_embedding_space = embedding_space.origin_embedding_space()
print(movie_embedding_space.shape)

director_tfidf, vocab = embedding_space.author_origin_tfidf()
print(director_tfidf.shape)

movie_space = embedding_space.origin_space()
print(movie_space.shape)

np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/User Target space/movie_space', movie_space)

# Target space
print('Target space')
book_embedding_space = embedding_space.target_embedding_space()
print(book_embedding_space.shape)

tfidf_author, _ = embedding_space.author_target_tfidf()
print(type(tfidf_author))
print(tfidf_author.shape)

book_space = embedding_space.target_space()
print(book_space.shape)

np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/User Target space/book_space', book_space)
# User space
print("User space")
users_embedding_space = embedding_space.build_users_profiles()
users_profile_matrix = np.vstack(users_embedding_space.values())
print(users_profile_matrix.shape)

np.save('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/User Target space/user_space', users_profile_matrix)
