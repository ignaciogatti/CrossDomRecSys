import pandas as pd
import numpy as np
import networkx as nx


class InfluenceGraph:

    def __init__(self, df_nodes, df_edges):
        self._type_dict = df_nodes.to_dict()
        self._color_dict = df_edges.to_dict()
        self._g_artist = nx.read_edgelist('/home/ignacio/Datasets/Graph analysis/artist_graph.edgelist', nodetype=str,
                                    data=False, delimiter=',')
        nx.set_node_attributes(self._g_artist, name='type', values=self._type_dict['type'])
        nx.set_edge_attributes(self._g_artist, name='color', values=self._color_dict['color'])


    def get_influence_graph(self):
        return self.g_artist_

