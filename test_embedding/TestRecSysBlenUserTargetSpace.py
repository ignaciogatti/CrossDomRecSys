import pandas as pd
import numpy as np
from cross_content_based_recSys.CrossContentBasedRecSys import CrossEmbContentBasedREcSys
from influence_graph.InfluenceGraph import InfluenceGraph



df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres_description.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating_description.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')
df_bu = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_bu.csv')
#Define influence graph
g_social = InfluenceGraph()

#origen_space
tfidf_matrix_origen = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/W2B/User Target space/movie_space.npy')

#target_space
tfidf_matrix_target = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/W2B/User Target space/book_space.npy')

#user_space
users_profiles_matrix = np.load('/home/ignacio/Datasets/Amazon/Data cleaned/Embedding/W2B/User Target space/user_space.npy')
users_id = df_movie_ratings['userId'].unique()
users_profiles = {}
idx = 0
for user_id in users_id:
    users_profiles[user_id] = users_profiles_matrix[idx].reshape(1,-1)
    idx += 1

#Define model
cross_content_model = CrossEmbContentBasedREcSys(
    df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen,
    user_profile=users_profiles, tfidf_matrix_target=tfidf_matrix_target, df_bu= df_bu, rating_matrix= df_movie_ratings,
    alpha=0.9)

#Recommendation
users_to_recommend =['A2EDZH51XHFA9B', 'A3UDYY6L2NH3JS', 'A2NJO6YE954DBH', 'AUM3YMZ0YRJE0', 'A17FLA8HQOFVIG',
                     'A20EEWWSFMZ1PN','AGEIT17HENDIS','ACIBQ6BQ6AWEV','A16QODENBJVUI1','A3KF4IP2MUS8QQ']

for user in users_to_recommend:
    df_recomendation = cross_content_model.recommend_items(user_id=user)
    df_recomendation.to_csv(
        '/home/ignacio/Datasets/Amazon/Data cleaned/Recommendation/Embedding/W2B/Blend/recommendation_user_to_target_space_'+user+'.csv', index='False')

print('Finished')