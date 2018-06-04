import pandas as pd
import scipy
from space_model.CreateVectorSpace import SimpleSpaceVector, OrigenTargetSpaceVector
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')

#Define influence graph
g_social = InfluenceGraph()

#Create vectors spaces
create_space_vector = OrigenTargetSpaceVector(g_social=g_social.get_influence_graph(), df_item_origen=df_ml_movies, df_item_target=df_bx_book)
(tfidf_matrix_origen, feature_names) = create_space_vector.item_origen_space()

print(tfidf_matrix_origen.shape)

tfidf_matrix_target = create_space_vector.define_target_space_from_origen()
tfidf_matrix_target = create_space_vector.build_target_space()


scipy.sparse.save_npz('/home/ignacio/Datasets/Amazon/Data cleaned/tfidf_matrix_target.npz', tfidf_matrix_target)
print(tfidf_matrix_target.shape)

'''
tfidf_matrix_target = create_space_vector._define_target_space_from_origen( df_bx_book )
tfidf_matrix_target = create_space_vector.build_target_space()


scipy.sparse.save_npz('/home/ignacio/Datasets/Graph analysis/tfidf_matrix_target.npz', tfidf_matrix_target)
print(tfidf_matrix_target.shape)

users_profiles = create_space_vector.build_users_profiles(df_movie_ratings)

#Define model
cross_content_model = CrossContentBasedRecSys(df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen, user_profile=users_profiles)
df_recomendation = cross_content_model.recommend_items(user_id=1)

print(df_recomendation)

#df_feature_names = pd.DataFrame( {'feature name':create_space_vector.tfidf_.get_feature_names(), 'feature mapped':create_space_vector.get_feature_space()} )
#df_feature_names.to_csv('/home/ignacio/Datasets/Graph analysis/feature_mapped.csv', index=False)
'''