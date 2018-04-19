import pandas as pd
import numpy as np
import networkx as nx
import scipy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize


class SimpleSpaceVector:

    def __init__(self, df_item = None):
        self._director_map = None
        self._df_item = df_item.copy()
        # Check stop words
        self._tfidf = None
        self._tfidf_matrix_origen = None
        self._users_profile = {}
        self._vocabulary_origen = None
        self._genres = None

    # Methods to get movie vector space (origen)
    def _create_director_map_(self):
        directors = self._df_item['director']
        directors = directors.str.split(r', ')
        directors_list = list(directors.values)
        directors_list = [d for director in directors_list for d in director]
        directors = pd.DataFrame(directors_list, columns=['director'])
        directors = directors.drop_duplicates()
        directors['normalize'] = directors['director'].str.lower()
        directors['normalize'] = directors['normalize'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        directors['normalize'] = directors['normalize'].str.replace(r'[\.\-\" ]', '')
        directors = directors.drop_duplicates(subset=['normalize'])
        directors = directors.set_index(['normalize'])
        self._director_map = directors

    def get_director_map_(self):
        directors_dict = self._director_map.to_dict()
        return directors_dict['director']

    def _define_vocabulary(self, genres):
        vocabulary_movie = list(self._director_map.index)
        genres = genres.str.split(' ')
        genres_list = list(genres.values)
        genres_list = [g for gen in genres_list for g in gen]
        genres = pd.DataFrame(genres_list, columns=['genre'])
        genres = genres.drop_duplicates()
        genres_list = list(genres['genre'].values)
        self._genres = genres_list
        vocabulary_movie.extend(genres_list)
        return vocabulary_movie

    def item_origen_space(self):
        self._create_director_map_()
        self._df_item['director'] = self._df_item['director'].str.lower()
        self._df_item['director'] = self._df_item['director'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        self._df_item['director'] = self._df_item['director'].str.replace(r'[\.\-\" ]', '')
        self._df_item['director'] = self._df_item['director'].str.replace(',', ' ')
        self._df_item['genres'] = self._df_item['genres'].str.replace('|', ' ')
        self._df_item['soap'] = self._df_item.apply(lambda x: x['genres'] + ' ' + x['director'], axis=1)
        self._vocabulary_origen = self._define_vocabulary( self._df_item['genres'])
        self._tfidf = TfidfVectorizer(vocabulary=self._vocabulary_origen)
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

    def __init__(self, g_social, df_item_origen = None, df_item_target = None):
        super().__init__(df_item=df_item_origen)
        self._df_item_target = df_item_target.copy()
        self._tfidf_matrix_target = None
        self._g_social = g_social


    def define_target_space_from_origen(self):
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
        iteracion = 0
        print('Iteracion ' + str(iteracion))
        for isbn in isbn_list:
            idx = self._df_item_target.index[self._df_item_target['ISBN'] == isbn].tolist()[0]
            df_book = self.define_target_space(idx)
            tfidf_book = df_book['tfidf'].as_matrix()
            tfidf_list.append( scipy.sparse.csr_matrix(tfidf_book) )
            iteracion += 1
            if iteracion%1000 == 0:
                print('Iteracion ' + str(iteracion))
        self._tfidf_matrix_target = scipy.sparse.vstack(tfidf_list)
        return self._tfidf_matrix_target



class TargetUserSpaceVector(SimpleSpaceVector):

    def __init__(self, g_social, df_item_origen = None,  df_item_target = None):
        super().__init__(df_item=df_item_origen)
        self._g_social = g_social
        self._df_item_target = df_item_target.copy()
        self._df_authors = None
        self._vocabulary_target = None
        self._tfidf_book = None
        self._tfidf_book_matrix = None


    def _create_author_map_(self):
        authors = self._df_item_target['Book-Author']
        authors_list = list(authors.unique())
        df_authors = pd.DataFrame(authors_list, columns=['author'])
        df_authors = df_authors.drop_duplicates()
        df_authors['normalize'] = df_authors['author'].str.lower()
        df_authors['normalize'] = df_authors['normalize'].str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('utf-8')
        df_authors['normalize'] = df_authors['normalize'].str.replace(r'[\.\-\" ]', '')
        df_authors = df_authors.drop_duplicates(subset=['normalize'])
        df_authors = df_authors.set_index(['normalize'])
        return df_authors


    def _define_vocabulary_target(self):
        vocabulary_book = list(self._df_authors.index)
        vocabulary_book.extend(self._genres)
        return vocabulary_book

    def item_target_space(self):
        self._df_authors = self._create_author_map_()
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].str.replace('|', ' ')
        self._df_item_target['Book-Author'] = self._df_item_target['Book-Author'].str.lower()
        self._df_item_target['Book-Author'] = self._df_item_target['Book-Author'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        self._df_item_target['Book-Author'] = self._df_item_target['Book-Author'].str.replace(r'[\.\-\" ]', '')
        self._df_item_target['soap'] = self._df_item_target.apply(lambda row: row['Book-Author'] + ' ' + row['common-shelves'], axis=1)
        self._vocabulary_target = self._define_vocabulary_target()
        self._tfidf_book = TfidfVectorizer(vocabulary=self._vocabulary_target)
        self._tfidf_book_matrix = self._tfidf_book.fit_transform(self._df_item_target['soap'])
        return (self._tfidf_book_matrix, self._tfidf_book.get_feature_names())

    def _tfidf_author(self, x, df_feature):
        df = df_feature.copy()
        df['shortest path'] = df.apply(lambda row: row['tfidf'] * 1. / (
        nx.shortest_path_length(G=self._g_social, source=x, target=row['feature name'])) if self._g_social.has_node(row['feature name']) and row['tfidf'] != 0.0 else 0.0, axis=1)
        return df['shortest path'].sum()

    def _get_user_item_space(self, user_profile_norm):
        df_feature_name = pd.DataFrame(self._tfidf.get_feature_names(), columns=['feature name'])
        df_user_item_space = pd.DataFrame(user_profile_norm)
        df_user_item_space = df_user_item_space.transpose()
        df_feature_name['tfidf'] = df_user_item_space[0]
        directors_dict = self.get_director_map_()
        df_feature_name['feature name'] = df_feature_name['feature name'].apply(lambda x: directors_dict[x] if x in directors_dict.keys() else x)
        return df_feature_name

    def _get_df_author_shortest_path(self):
        df_authors_shortest_path = pd.DataFrame(self._df_authors['author'], columns=['author'])
        df_authors_shortest_path = df_authors_shortest_path.reset_index(drop=True)
        return df_authors_shortest_path

    def build_user_profile(self, df_ratings, userId):
        user_profile_norm = super().build_user_profile(df_ratings, userId)
        df_user_item_space = self._get_user_item_space(user_profile_norm)
        df_auhtors_shortest_path = self._get_df_author_shortest_path()
        df_auhtors_shortest_path['tfidf'] = df_auhtors_shortest_path['author'].apply(self._tfidf_author, df_feature=df_user_item_space)
        df_user_item_space_without_directors = df_user_item_space[~df_user_item_space['feature name'].isin(self._director_map['director'])]
        df_auhtors_shortest_path = df_auhtors_shortest_path.rename(index=str, columns={'author': 'feature name'})
        df_user_item_space_book = pd.concat([df_auhtors_shortest_path, df_user_item_space_without_directors])
        return df_user_item_space_book

    def build_users_profiles(self, df_ratings):
        users_id = df_ratings['userId'].unique()
        for user_id in users_id:
            df_user_item_space_book = self.build_user_profile(df_ratings, user_id)
            user_item_space_book_matrix = df_user_item_space_book['tfidf'].as_matrix()
            self.build_users_profile[user_id] = user_item_space_book_matrix.reshape((1,-1))

        return self._users_profile







