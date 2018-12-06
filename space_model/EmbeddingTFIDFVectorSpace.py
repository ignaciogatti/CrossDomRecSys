import pandas as pd
import numpy as np
import networkx as nx
from space_model.EmbeddingVectorSpace import EmbeddingVectorSpace
from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingTFIDFauthorSpace(EmbeddingVectorSpace):

    def __init__(self,df_item_origin=None, df_ratings = None):
        super().__init__(df_ratings=df_ratings, df_item_origin=df_item_origin)
        self._director_map = None
        self._tfidf_origin = None
        self._tfidf_matrix_origen = None

    # Methods to get author space (origen)
    def _create_director_map_(self):
        directors = self._df_item_origin['director']
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

    def author_origin_tfidf(self):
        self._create_director_map_()
        self._df_item_origin['director tfidf'] = self._df_item_origin['director'].str.lower()
        self._df_item_origin['director tfidf'] = self._df_item_origin['director tfidf'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        self._df_item_origin['director tfidf'] = self._df_item_origin['director tfidf'].str.replace(r'[\.\-\" ]', '')
        self._df_item_origin['director tfidf'] = self._df_item_origin['director tfidf'].str.replace(',', ' ')
        self._tfidf_origin = TfidfVectorizer(vocabulary=list(self._director_map.index))
        self._tfidf_matrix_origen = self._tfidf_origin.fit_transform(self._df_item_origin['director tfidf'])
        return (self._tfidf_matrix_origen, self._tfidf_origin.get_feature_names())

    def origin_space(self):
        return np.concatenate((self._origin_embeddings, self._tfidf_matrix_origen.toarray()), axis=1)

    # Methods to get user vector space
    def concat_user_profile(self, userId):
        items_id = (self._df_ratings[self._df_ratings['userId'] == userId]['movieId']).values
        user_rating_weigth = (self._df_ratings[self._df_ratings['userId'] == userId]['rating']).values.reshape(-1, 1)
        embedding_space = super().build_user_profile( items_id, user_rating_weigth, self._origin_embeddings)
        author_tfidf_space =  super().build_user_profile( items_id, user_rating_weigth, self._tfidf_matrix_origen.toarray())
        return np.concatenate((embedding_space, author_tfidf_space), axis=1)

    def build_users_profiles(self):
        users_id = self._df_ratings['userId'].unique()
        for user_id in users_id:
            self._users_profile[user_id] = self.concat_user_profile(user_id)
        return self._users_profile


class EmbeddingTFIDFauthorTargetSpace(EmbeddingTFIDFauthorSpace):

    def __init__(self, df_item_origin=None, df_item_target=None, df_ratings = None, g_social=None):
        super().__init__(df_item_origin=df_item_origin, df_ratings=df_ratings)
        self._df_item_target = df_item_target.copy()
        self._target_embeddings = None
        self._tfidf_target_map_matrix = None
        self._g_social = g_social

    def target_embedding_space(self):
        self._df_item_target['description'] = self._df_item_target['description'].fillna('no description')
        self._df_item_target['description'] = self._df_item_target['description'].apply(self.process_description)
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].str.replace('|', ' ')
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].apply(self.process_description)
        self._df_item_target['Book-Author emb'] = self._df_item_target['Book-Author'].apply(self.process_authors)
        self._df_item_target['soap'] = self._df_item_target['description'] + self._df_item_target['common-shelves'] + self._df_item_target['Book-Author emb']
        target_embeddings = self._df_item_target['soap'].apply(super().get_w2vec)
        self._target_embeddings = np.stack(target_embeddings)
        return self._target_embeddings

    def _get_shortest_path(self, author):
        #check this function
        df_map_to_dir = pd.DataFrame(self._tfidf_origin.get_feature_names(), columns=['director'])
        director_map = super().get_director_map_()
        tfidf_map = df_map_to_dir['director'].apply(lambda x: 1 / nx.shortest_path_length(
            G=self._g_social, source=director_map[x], target=author) if self._g_social.has_node(director_map[x]) else 0)
        return tfidf_map.values.reshape((-1,))

    def get_tfidf_mapped(self):
        tfidf_map = self._df_item_target['Book-Author'].apply(self._get_shortest_path)
        self._tfidf_target_map_matrix = np.stack(tfidf_map)
        return self._tfidf_target_map_matrix

    def target_space(self):
        return np.concatenate((self._target_embeddings, self._tfidf_target_map_matrix), axis=1)


class EmbeddingTFIDFauthorUserTargetSpace(EmbeddingTFIDFauthorSpace):

    def __init__(self, df_item_origin=None, df_item_target=None, df_ratings = None, g_social=None):
        super().__init__(df_item_origin=df_item_origin, df_ratings=df_ratings)
        self._df_item_target = df_item_target.copy()
        self._target_embeddings = None
        self._tfidf_matrix_target = None
        self._tfidf_target = None
        self._g_social = g_social
        self._df_authors = None

    def target_embedding_space(self):
        self._df_item_target['description'] = self._df_item_target['description'].fillna('no description')
        self._df_item_target['description'] = self._df_item_target['description'].apply(self.process_description)
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].str.replace('|', ' ')
        self._df_item_target['common-shelves'] = self._df_item_target['common-shelves'].apply(self.process_description)
        self._df_item_target['Book-Author emb'] = self._df_item_target['Book-Author'].apply(self.process_authors)
        self._df_item_target['soap'] = self._df_item_target['description'] + self._df_item_target['common-shelves'] + self._df_item_target['Book-Author emb']
        target_embeddings = self._df_item_target['soap'].apply(super().get_w2vec)
        self._target_embeddings = np.stack(target_embeddings)
        return self._target_embeddings

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

    def author_target_tfidf(self):
        self._df_authors = self._create_author_map_()
        self._df_item_target['Book-Author tfidf'] = self._df_item_target['Book-Author'].str.lower()
        self._df_item_target['Book-Author tfidf'] = self._df_item_target['Book-Author tfidf'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        self._df_item_target['Book-Author tfidf'] = self._df_item_target['Book-Author tfidf'].str.replace(r'[\.\-\" ]', '')
        self._tfidf_target = TfidfVectorizer(vocabulary=list(self._df_authors.index))
        self._tfidf_matrix_target = self._tfidf_target.fit_transform(self._df_item_target['Book-Author tfidf'])
        return (self._tfidf_matrix_target, self._tfidf_target.get_feature_names())

    def target_space(self):
        return np.concatenate((self._target_embeddings, self._tfidf_matrix_target.toarray()), axis=1)

    def _tfidf_author(self, x, df_feature):
        df = df_feature.copy()
        df['shortest path'] = df.apply(lambda row: row['tfidf'] * 1. / (
        nx.shortest_path_length(G=self._g_social, source=x, target=row['director'])) if self._g_social.has_node(
            row['director']) and row['tfidf'] != 0.0 else 0.0, axis=1)
        return df['shortest path'].sum()

    def _get_user_item_space(self, user_profile_norm):
        df_feature_name = self._director_map.copy().reset_index(drop=True)
        df_feature_name['tfidf'] = user_profile_norm
        return df_feature_name

    def _get_df_author_shortest_path(self):
        df_authors_shortest_path = pd.DataFrame(self._df_authors['author'], columns=['author'])
        df_authors_shortest_path = df_authors_shortest_path.reset_index(drop=True)
        return df_authors_shortest_path

    def map_user_profile(self, user_profile_norm):
        df_user_item_space = self._get_user_item_space(user_profile_norm)
        df_auhtors_shortest_path = self._get_df_author_shortest_path()
        tfidif_user = df_auhtors_shortest_path['author'].apply(self._tfidf_author, df_feature=df_user_item_space)
        return tfidif_user.values.reshape((1,-1))

    # Methods to get user vector space
    def concat_user_profile(self, userId):
        items_id = (self._df_ratings[self._df_ratings['userId'] == userId]['movieId']).values
        user_rating_weigth = (self._df_ratings[self._df_ratings['userId'] == userId]['rating']).values.reshape(-1, 1)
        embedding_space = super().build_user_profile( items_id, user_rating_weigth, self._origin_embeddings)
        author_tfidf_space =  super().build_user_profile( items_id, user_rating_weigth, self._tfidf_matrix_origen.toarray())
        author_tfidf_space = self.map_user_profile(author_tfidf_space.reshape((-1,)))
        return np.concatenate((embedding_space, author_tfidf_space), axis=1)

