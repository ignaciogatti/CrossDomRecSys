import pandas as pd
import scipy
from space_model.CreateVectorSpace import SimpleSpaceVector, OrigenTargetSpaceVector
from cross_content_based_recSys.CrossContentBasedRecSys import ContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres.csv')
df_bu = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_bu.csv')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')

#Create vectors spaces
create_space_vector = SimpleSpaceVector(df_item=df_ml_movies)
tfidf_matrix, _ = create_space_vector.item_origen_space()
users_profile = create_space_vector.build_users_profiles(df_movie_ratings)

#Define CB model
cb = ContentBasedRecSys( df_items_origen= df_ml_movies,user_profile=users_profile,
                         tfidf_matrix_origen=tfidf_matrix, rating_matrix=df_movie_ratings, df_bu=df_bu)

result = cb.recommend_items(user_id='AGEIT17HENDIS')

result.to_csv('/home/ignacio/Datasets/Amazon/Data cleaned/Recommendation/recommendation_rating_proof.csv', index=False)
print(result.head())