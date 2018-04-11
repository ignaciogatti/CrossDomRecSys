import pandas as pd
import numpy as np
import networkx as nx


class InfluenceGraph:

    def __init__(self):
        #Get nodes
        df_nodes = pd.read_csv('/home/ignacio/Datasets/Graph analysis/artists_nodes.csv')
        df_nodes = df_nodes.set_index('Artists')
        self._type_dict = df_nodes.to_dict()
        #Get edges
        df_edges = pd.read_csv('/home/ignacio/Datasets/Graph analysis/artists_edges.csv')
        df_edges = df_edges.set_index(['Node 1', 'Node 2'])
        self._color_dict = df_edges.to_dict()
        #Create graph
        self._g_artist = nx.read_edgelist('/home/ignacio/Datasets/Graph analysis/artist_graph.edgelist', nodetype=str,
                                    data=False, delimiter=',')
        nx.set_node_attributes(self._g_artist, name='type', values=self._type_dict['type'])
        nx.set_edge_attributes(self._g_artist, name='color', values=self._color_dict['color'])


    def get_influence_graph(self):
        return self._g_artist

