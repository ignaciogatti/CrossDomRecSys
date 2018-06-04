import pandas as pd
import scipy
from space_model.CreateVectorSpace import SimpleSpaceVector, OrigenTargetSpaceVector
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys, ContentBasedRecSys
from influence_graph.InfluenceGraph import InfluenceGraph


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_meta_valid_genres.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/book_meta_valid_shelves_rating.csv')
df_bx_book['common-shelves'] = df_bx_book['common-shelves'].fillna('')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/ratings_movie_intersect_ii.csv')
df_bu = pd.read_csv('/home/ignacio/Datasets/Amazon/Data cleaned/movie_bu.csv')


#Define influence graph
g_social = InfluenceGraph()

#Create vectors spaces
create_space_vector = OrigenTargetSpaceVector(g_social=g_social.get_influence_graph(), df_item_origen=df_ml_movies, df_item_target=df_bx_book)
#origen_space
(tfidf_matrix_origen, feature_names) = create_space_vector.item_origen_space()

#target_space
tfidf_matrix_target = scipy.sparse.load_npz('/home/ignacio/Datasets/Amazon/Data cleaned/tfidf_matrix_target.npz')

#user_space
users_profiles = create_space_vector.build_users_profiles(df_movie_ratings)

#Define model
cross_content_model = CrossContentBasedRecSys(
    df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen,
    user_profile=users_profiles, tfidf_matrix_target=tfidf_matrix_target, df_bu=df_bu, rating_matrix=df_movie_ratings)

#Recommendation
users_to_recommend =['A2EDZH51XHFA9B', 'A3UDYY6L2NH3JS', 'A2NJO6YE954DBH', 'AUM3YMZ0YRJE0', 'A17FLA8HQOFVIG',
                     'A20EEWWSFMZ1PN','AGEIT17HENDIS','ACIBQ6BQ6AWEV','A16QODENBJVUI1','A3KF4IP2MUS8QQ']

for user_id in users_to_recommend:
    df_recomendation = cross_content_model.recommend_items(user_id=user_id)
    df_recomendation.to_csv(
        '/home/ignacio/Datasets/Amazon/Data cleaned/Recommendation/recommendation_book_to_origen_space_'+user_id+'.csv',
        index='False')

print('Finished')

#df_feature_names = pd.DataFrame( {'feature name':create_space_vector.tfidf_.get_feature_names(), 'feature mapped':create_space_vector.get_feature_space()} )
#df_feature_names.to_csv('/home/ignacio/Datasets/Graph analysis/feature_mapped.csv', index=False)
