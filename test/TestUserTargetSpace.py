import pandas as pd
import scipy
from space_model.CreateVectorSpace import SimpleSpaceVector, OrigenTargetSpaceVector, TargetUserSpaceVector
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-valid-movies.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Graph analysis/bx-valid-book.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-ratings.csv')

#Define influence graph
g_social = InfluenceGraph()

#Create vectors spaces
create_space_vector = TargetUserSpaceVector(g_social=g_social.get_influence_graph(), df_item_origen=df_ml_movies, df_item_target=df_bx_book)
(tfidf_matrix_origen, feature_names) = create_space_vector.item_origen_space()

(tfidf_matrix_target, feature_names_target) = create_space_vector.item_target_space()

df_user_space = create_space_vector.build_user_profile(df_ratings=df_movie_ratings, userId=1)
#tfidf_matrix_target = create_space_vector.build_target_space()


print(df_user_space.head())
