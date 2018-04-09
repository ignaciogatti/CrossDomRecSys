import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class Create_Space_Vector:

    def __init__(self):
        self._director_map = {}
        self._df_item = None
        self._df_item_target = None
        #Check stop words
        self._tfidf = TfidfVectorizer(stop_words='english')
        self._tfidf_matrix_origen = None
        self._tfidf_matrix_target = None
        self._users_profile = {}


# Methods to get movie vector space (origen)
    def get_director_map_(self):

        directors = self._df_item['director']
        directors = directors.str.split(r', ')
        directors_list = list(directors.values)
        directors_list = [d for director in directors_list for d in director]
        directors = pd.DataFrame(directors_list, columns=['director'])
        directors = directors.drop_duplicates()
        directors['normalize'] = directors['director'].str.lower()
        directors['normalize'] = directors['normalize'].str.replace(' ', '')
        directors = directors.set_index(['normalize'])
        directors_dict = directors.to_dict()
        return directors_dict['director']


    def item_origen_space(self, df_movie):

        self._df_item = df_movie.copy()
        self._director_map = self.get_director_map_()
        self._df_item['director'] = self._df_item['director'].str.lower()
        self._df_item['director'] = self._df_item['director'].str.replace(r'[\.\- ]', '')
        self._df_item['director'] = self._df_item['director'].str.replace(',', ' ')
        self._df_item['genres'] = df_movie['genres'].str.replace('|', ' ')
        self._df_item['soap'] = self._df_item.apply(lambda x: x['genres'] + ' ' + x['director'], axis=1)
        self._tfidf_matrix_origen = self._tfidf.fit_transform(self._df_item['soap'])
        return (self._tfidf_matrix_origen, self._tfidf.get_feature_names())


    def get_feature_space(self):

        feature_names = self._tfidf.get_feature_names()
        feature_names_mapped = [ self._director_map[feat] if feat in self._director_map.keys() else feat for feat in feature_names ]
        return feature_names_mapped



# Methods to get book vector space (target)
    def item_target_space(self, df_book):
        self._df_item_target = df_book.copy()
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].str.replace('|', ' ')
        self._tfidf_matrix_target = self._tfidf.transform(self._df_item_target['common-shelves'])
        return (self._tfidf_matrix_target, self._tfidf.get_feature_names())


    def map_target_to_origen_space(self):
        return None



# Methods to get user vector space
    def get_item_profile(self, item_id):

        idx = self._df_item.index[self._df_item['movieId']==item_id].tolist()[0]
        item_profile = self._tfidf_matrix_origen[idx:idx + 1]
        return item_profile


    def get_item_profiles(self, ids):

        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles


    def build_user_profile(self, df_ratings, userId):
        items_id = (df_ratings[df_ratings['userId'] == userId]['movieId']).values
        user_item_profiles = self.get_item_profiles(items_id)
        user_rating_weigth = (df_ratings[df_ratings['userId'] == userId]['rating']).values.reshape(-1,1)
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_rating_weigth), axis=0) / np.sum(user_rating_weigth)
        user_profile_norm = normalize(user_item_strengths_weighted_avg)
        return user_profile_norm


    def build_users_profiles(self, df_ratings):
        users_id = df_ratings['userId'].unique()
        for user_id in users_id:
            self._users_profile[user_id] = self.build_user_profile(df_ratings, user_id)
        return self._users_profile






