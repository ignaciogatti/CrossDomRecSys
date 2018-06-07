from gensim.models import Word2Vec, KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords, preprocess_string, strip_multiple_whitespaces, strip_punctuation, strip_tags
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import numpy as np
import scipy

class EmbeddingVectorSpace:

    def __init__(self, df_item_origin=None, df_ratings = None):
        self._word2vect_model = KeyedVectors.load_word2vec_format(
            '/home/ignacio/Datasets/Word Embeddings Pretrained Vectors/Word2Vec/GoogleNews-vectors-negative300.bin',
            binary=True)
        self._word2vect_model.init_sims(replace=True)
        self._df_item_origin = df_item_origin.copy()
        self._users_profile = {}
        self._origin_embeddings = None
        self._df_ratings = df_ratings

    #Methods to get embeddings
    def process_description(self, description):
        description = description.lower()
        description = strip_punctuation(description)
        description = strip_tags(description)
        description = strip_multiple_whitespaces(description)
        description = description.split(' ')
        description_clean = list(filter(lambda x: x in self._word2vect_model.vocab, description))
        return description_clean

    def process_authors(self, authors):
        authors = authors.split(' ')
        author_clean = list(filter(lambda x: x in self._word2vect_model.vocab, authors))
        return author_clean

    def get_w2vec(self, soap):
        # vector space of embeddings
        w2v = np.zeros((300,))
        for s in soap:
            w2v += self._word2vect_model.wv[s]
        return w2v / len(soap)

    def origin_embedding_space(self):
        self._df_item_origin['description'] = self._df_item_origin['description'].apply(self.process_description)
        self._df_item_origin['genres'] = self._df_item_origin['genres'].str.replace('|', ' ')
        self._df_item_origin['genres'] = self._df_item_origin['genres'].apply(self.process_description)
        self._df_item_origin['director emb'] = self._df_item_origin['director'].str.replace(',', '')
        self._df_item_origin['director emb'] = self._df_item_origin['director emb'].apply(self.process_authors)
        self._df_item_origin['soap'] = self._df_item_origin['description'] + self._df_item_origin['genres'] + self._df_item_origin['director emb']
        origin_embeddings = self._df_item_origin['soap'].apply(self.get_w2vec)
        self._origin_embeddings = np.stack(origin_embeddings)
        return self._origin_embeddings


    # Methods to get user vector space
    def get_item_profile(self, item_id, matrix):
        idx = self._df_item_origin.index[self._df_item_origin['movieId'] == item_id].tolist()[0]
        item_profile = matrix[idx:idx + 1]
        return item_profile.reshape((-1,))

    def get_item_profiles(self, ids, matrix):
        item_profiles_list = [self.get_item_profile(x, matrix) for x in ids]
        item_profiles = np.stack(item_profiles_list)
        return item_profiles

    def build_user_profile(self, items_id, user_rating_weigth, matrix):
        user_item_profiles = self.get_item_profiles(items_id, matrix)
        user_item_strengths_weighted_avg = np.sum(np.multiply(user_item_profiles, user_rating_weigth), axis=0) / np.sum(
            user_rating_weigth)
        user_profile_norm = normalize(user_item_strengths_weighted_avg.reshape((1,-1)))
        return user_profile_norm

    def build_users_profiles(self):
        users_id = self._df_ratings['userId'].unique()
        for user_id in users_id:
            items_id = (self._df_ratings[self._df_ratings['userId'] == user_id]['movieId']).values
            user_rating_weigth = (self._df_ratings[self._df_ratings['userId'] == user_id]['rating']).values.reshape(-1,1)
            self._users_profile[user_id] = self.build_user_profile( items_id, user_rating_weigth, self._origin_embeddings)
        return self._users_profile


class EmbeddingVectorTargetSpace(EmbeddingVectorSpace):

    def __init__(self, df_item_origin=None, df_item_target=None, df_ratings = None):
        super().__init__(df_item_origin=df_item_origin, df_ratings=df_ratings)
        self._df_item_target = df_item_target.copy()
        self._target_embeddings = None


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
