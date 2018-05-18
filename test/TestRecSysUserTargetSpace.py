import pandas as pd
import numpy as np
from space_model.CreateVectorSpace import SimpleSpaceVector, TargetUserSpaceVector
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys, ContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-valid-movies.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Graph analysis/bx-valid-book.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-ratings.csv')

#Define influence graph
g_social = InfluenceGraph()

#Create vectors spaces
create_space_vector = TargetUserSpaceVector(g_social=g_social.get_influence_graph(), df_item_origen=df_ml_movies, df_item_target=df_bx_book)
#origen_space
(tfidf_matrix_origen, feature_names_movie) = create_space_vector.item_origen_space()

#target_space
#tfidf_matrix_target = create_space_vector._define_target_space_from_origen( df_bx_book )
(tfidf_matrix_target, feature_names_book) = create_space_vector.item_target_space()

#user_space
users_profiles_matrix = np.load('/home/ignacio/Datasets/Graph analysis/user_book_space.npy')
users_id = df_movie_ratings['userId'].unique()
users_profiles = {}
idx = 0
for user_id in users_id:
    users_profiles[user_id] = users_profiles_matrix[idx].reshape(1,-1)
    idx += 1

#Define model
cross_content_model = CrossContentBasedRecSys(
    df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen,
    user_profile=users_profiles, tfidf_matrix_target=tfidf_matrix_target)

#Recommendation
df_recomendation = cross_content_model.recommend_items(user_id=1)
df_recomendation.to_csv('/home/ignacio/Datasets/Graph analysis/Recommendation/recommendation_user_to_target_space.csv', index='False')

print(df_recomendation.head())
