import pandas as pd
from space_model.Create_Space_Vector import Create_Space_Vector
from cross_content_based_recSys.CrossContentBasedRecSys import CrossContentBasedRecSys


df_ml_movies = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-valid-movies.csv')
df_bx_book = pd.read_csv('/home/ignacio/Datasets/Graph analysis/bx-valid-book.csv')
df_movie_ratings = pd.read_csv('/home/ignacio/Datasets/Graph analysis/ml-ratings.csv')

#Create vectors spaces
create_space_vector = Create_Space_Vector()
(tfidf_matrix_origen, feature_names) = create_space_vector.item_origen_space(df_ml_movies)
users_profiles = create_space_vector.build_users_profiles(df_movie_ratings)

#Define model
cross_content_model = CrossContentBasedRecSys(df_items_origen=df_ml_movies, df_items_target=df_bx_book ,tfidf_matrix_origen= tfidf_matrix_origen, user_profile=users_profiles)
df_recomendation = cross_content_model.recommend_items(user_id=1)




#df_user_profile = pd.DataFrame(user_profile)

#df_user_profile = df_user_profile.transpose()
#df_user_profile['feature mapped'] = create_space_vector.get_feature_space()
#df_user_profile.to_csv('/home/ignacio/Datasets/Graph analysis/user_profile_example.csv', index=False)

print(df_recomendation)

#df_feature_names = pd.DataFrame( {'feature name':create_space_vector.tfidf_.get_feature_names(), 'feature mapped':create_space_vector.get_feature_space()} )
#df_feature_names.to_csv('/home/ignacio/Datasets/Graph analysis/feature_mapped.csv', index=False)

