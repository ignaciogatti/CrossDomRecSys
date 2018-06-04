import pandas as pd
import numpy as np
from space_model.EmbeddingVectorSpace import EmbeddingVectorSpace, EmbeddingVectorTargetSpace
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys, ContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres_description.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating_description.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')
df_bu = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_bu.csv')



#embedding_space = EmbeddingVectorSpace(df_item_origin=df_ml_movies, df_ratings=df_movie_ratings)

embedding_space = EmbeddingVectorTargetSpace(df_item_origin=df_ml_movies, df_item_target=df_bx_book, df_ratings=df_movie_ratings)

(tfidf_author, vocab) = embedding_space.author_target_tfidf()

print(vocab)

print(embedding_space.get_director_map_().head())
'''
(tfidf_author, vocab) = embedding_space.get_author_tfidf()

print(vocab)

movie_embedding_space = embedding_space.origin_embedding_space()

users_embedding_space = embedding_space.build_users_profiles()

users_profile_matrix = np.vstack(users_embedding_space.values())

print(users_profile_matrix.shape)

np.save('/home/ignacio/Datasets/Amazon/Data cleaned/user_embedding_space', users_profile_matrix)

'''
