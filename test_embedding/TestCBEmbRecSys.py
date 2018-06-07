import pandas as pd
import numpy as np
from cross_content_based_recSys.CrossContentBasedRecSys import ContentBasedRecSys, CrossEmbContentBasedREcSys
from influence_graph.InfluenceGraph import InfluenceGraph

df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres_description.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating_description.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')
df_bu = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_bu.csv')
#Define influence graph
g_social = InfluenceGraph()

#origen_space
tfidf_matrix_origen = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/Target to Origin space/movie_space.npy')

#target_space
tfidf_matrix_target = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/Target to Origin space/book_space.npy')

#user_space
users_profiles_matrix = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/Target to Origin space/user_space.npy')
users_id = df_movie_ratings['userId'].unique()
users_profiles = {}
idx = 0
for user_id in users_id:
    users_profiles[user_id] = users_profiles_matrix[idx].reshape(1,-1)
    idx += 1

#Define model
cross_content_model = CrossEmbContentBasedREcSys(
    df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen,
    user_profile=users_profiles, tfidf_matrix_target=tfidf_matrix_target, df_bu= df_bu, rating_matrix= df_movie_ratings)


result = cross_content_model.recommend_items(user_id='AGEIT17HENDIS')

result.to_csv('/home/ignacio/Datasets/Amazon/Data cleaned/Recommendation/Embedding/recommendation_rating_proof.csv', index=False)
print(result.head())