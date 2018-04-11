import pandas as pd
import numpy as np
import networkx as nx
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class SimpleSpaceVector:

    def __init__(self):
        self._director_map = None
        self._df_item = None
        # Check stop words
        self._tfidf = TfidfVectorizer(stop_words='english')
        self._tfidf_matrix_origen = None
        self._users_profile = {}

    # Methods to get movie vector space (origen)
    def _create_director_map_(self):
        directors = self._df_item['director']
        directors = directors.str.split(r', ')
        directors_list = list(directors.values)
        directors_list = [d for director in directors_list for d in director]
        directors = pd.DataFrame(directors_list, columns=['director'])
        directors = directors.drop_duplicates()
        directors['normalize'] = directors['director'].str.lower()
        directors['normalize'] = directors['normalize'].str.replace(' ', '')
        directors = directors.set_index(['normalize'])
        self._director_map = directors

    def get_director_map_(self):
        directors_dict = self._director_map.to_dict()
        return directors_dict['director']

    def item_origen_space(self, df_movie):
        self._df_item = df_movie.copy()
        self._create_director_map_()
        self._df_item['director'] = self._df_item['director'].str.lower()
        self._df_item['director'] = self._df_item['director'].str.replace(r'[\.\- ]', '')
        self._df_item['director'] = self._df_item['director'].str.replace(',', ' ')
        self._df_item['genres'] = df_movie['genres'].str.replace('|', ' ')
        self._df_item['soap'] = self._df_item.apply(lambda x: x['genres'] + ' ' + x['director'], axis=1)
        self._tfidf_matrix_origen = self._tfidf.fit_transform(self._df_item['soap'])
        return (self._tfidf_matrix_origen, self._tfidf.get_feature_names())

    def get_feature_space(self):
        feature_names = self._tfidf.get_feature_names()
        directors_dict = self.get_director_map_()
        feature_names_mapped = [directors_dict[feat] if feat in directors_dict.keys() else feat for feat in
                                feature_names]
        return feature_names_mapped

    # Methods to get user vector space
    def get_item_profile(self, item_id):
        idx = self._df_item.index[self._df_item['movieId'] == item_id].tolist()[0]
        item_profile = self._tfidf_matrix_origen[idx:idx + 1]
        return item_profile

    def get_item_profiles(self, ids):
        item_profiles_list = [self.get_item_profile(x) for x in ids]
        item_profiles = scipy.sparse.vstack(item_profiles_list)
        return item_profiles

    def build_user_profile(self, df_ratings, userId):
        items_id = (df_ratings[df_ratings['userId'] == userId]['movieId']).values
        user_item_profiles = self.get_item_profiles(items_id)
        user_rating_weigth = (df_ratings[df_ratings['userId'] == userId]['rating']).values.reshape(-1, 1)
        user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_rating_weigth), axis=0) / np.sum(
            user_rating_weigth)
        user_profile_norm = normalize(user_item_strengths_weighted_avg)
        return user_profile_norm

    def build_users_profiles(self, df_ratings):
        users_id = df_ratings['userId'].unique()
        for user_id in users_id:
            self._users_profile[user_id] = self.build_user_profile(df_ratings, user_id)
        return self._users_profile


class OrigenTargetSpaceVector(SimpleSpaceVector):

    def __init__(self, g_social):
        super().__init__()
        self._df_item_target = None
        self._tfidf_matrix_target = None
        self._g_social = g_social


    def _define_target_space_from_origen(self, df_book):
        self._df_item_target = df_book.copy()
        self._df_item_target['soap'] = self._df_item_target['common-shelves'].str.replace('|', ' ')
        self._tfidf_matrix_target = self._tfidf.transform(self._df_item_target['soap'])
        return self._tfidf_matrix_target


    def _get_shortest_path(self, author):
        #check this function
        df_directors_influence = self._director_map.copy()
        df_directors_influence = df_directors_influence.reset_index(drop=True)
        df_directors_influence['short path'] = df_directors_influence['director'].apply(
            lambda x: nx.shortest_path_length(G=self._g_social, source=x,
                                              target=author) if self._g_social.has_node(x) else -1)
        df_directors_influence = df_directors_influence.set_index('director')
        return df_directors_influence


    def define_target_space(self, idx):
        tfidf_book = self._tfidf_matrix_target[ idx:idx + 1 ]
        df_book_tfidf = pd.DataFrame(tfidf_book.todense())
        df_book_tfidf = df_book_tfidf.transpose()
        directors_dict = self.get_director_map_()
        df_book = pd.DataFrame(self._tfidf.get_feature_names(), columns=['feature name'])
        df_book['feature name'] = df_book['feature name'].apply(
            lambda x: directors_dict[x] if x in directors_dict.keys() else x)
        df_book['tfidf'] = df_book_tfidf[0]
        df_director_influence = self._get_shortest_path( self._df_item_target.iloc[idx]['Book-Author'])
        df_book['tfidf'] = df_book.apply(
            lambda row: 1.0 / (df_director_influence.loc[row['feature name']]['short path']) if (
            row['feature name'] in df_director_influence.index) else row['tfidf'], axis=1)
        df_book['tfidf'] = df_book['tfidf'].replace(-1.0, 0.0)
        return df_book


    def build_target_space(self):
        isbn_list = self._df_item_target['ISBN'].unique()
        tfidf_list =[]
        for isbn in isbn_list:
            idx = self._df_item_target.index[self._df_item_target['ISBN'] == isbn].tolist()[0]
            df_book = self.define_target_space(idx)
            tfidf_book = df_book['tfidf'].as_matrix()
            tfidf_list.append( scipy.sparse.coo_matrix(tfidf_book) )
        self._tfidf_matrix_target = scipy.sparse.vstack(tfidf_list)
        return self._tfidf_matrix_target


