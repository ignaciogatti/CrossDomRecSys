import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import  cosine_similarity, linear_kernel


class ContentBasedRecSys:

    MODEL_NAME = 'Content-Based'

    def __init__(self, df_items_origen=None, user_profile=None, tfidf_matrix_origen=None):
        self._df_items_origen = df_items_origen.copy()
        self._user_profile = user_profile
        self._tfidf_matrix_origen = tfidf_matrix_origen


    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=100):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self._user_profile[person_id], self._tfidf_matrix_origen)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self._df_items_origen.iloc[i]['movieId'], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items

    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        df_recommendation = pd.DataFrame(similar_items_filtered, columns=['id', 'similarity']).head(topn)
        df_recommendation = pd.merge( df_recommendation, self._df_items_origen, how='left', right_on='movieId', left_on='id' )
        print( df_recommendation.columns )
        df_recommendation = df_recommendation[['movieId', 'title', 'year', 'genres', 'director', 'similarity']]

        return df_recommendation


class CrossContentBasedRecSys(ContentBasedRecSys):

    MODEL_NAME = 'Cross-Content-Based'

    def __init__(self, df_items_origen=None, df_items_target=None, user_profile=None, tfidf_matrix_origen=None, tfidf_matrix_target=None,  g_artist_influence=None):
        super().__init__(df_items_origen=df_items_origen, user_profile=user_profile, tfidf_matrix_origen=tfidf_matrix_origen)
        self._df_items_target = df_items_target.copy()
        self._tfidf_matrix_target = tfidf_matrix_target
        self._g_artist_influence = g_artist_influence


    def get_model_name(self):
        return self.MODEL_NAME

    def _get_similar_items_to_user_profile(self, person_id, topn=100):
        # Computes the cosine similarity between the user profile and all item profiles
        cosine_similarities = cosine_similarity(self._user_profile[person_id], self._tfidf_matrix_target)
        # Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        # Sort the similar items by similarity
        similar_items = sorted([(self._df_items_target.iloc[i]['ISBN'], cosine_similarities[0, i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items


    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        # Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))

        df_recommendation = pd.DataFrame(similar_items_filtered, columns=['id', 'similarity']).head(topn)
        df_recommendation = pd.merge( df_recommendation, self._df_items_target, how='left', right_on='ISBN', left_on='id' )
        print( df_recommendation.columns )
        df_recommendation = df_recommendation[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'similarity']]

        return df_recommendation

        