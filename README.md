# CrossDomRecSys
![Cross Domain Approach ](https://github.com/ignaciogatti/CrossDomRecSys/blob/master/images/Approach.jpg)
The idea of this project is to define a Recommender System that takes knowledge from a source domian (movies) and apply to a target one (books).
In other words, the objective is to take advantage of the source domain data -this is a well known domain- to attack the cold-start problem in the target domain.

[Here](http://sedici.unlp.edu.ar/bitstream/handle/10915/73027/Documento_completo.pdf-PDFA.pdf?sequence=1&isAllowed=y) you can find a detailed description of the idea (the paper is in spanish)

# Project detail
One of the challenges of this project was to model movies and books in the same space. In order to solve it, we decide to model using word embedding. Basically, we take the metadata of each item and map it to a word embedding. To develop, we prove different configurations of  pre-trained models (Word2Vect and GLoVe), using [GenSim library](https://radimrehurek.com/gensim/about.html).

To link books and movies, we decided to generate an influence graph that connect authors. This graph was extracted from [DBpedia Ontology](https://wiki.dbpedia.org/services-resources/ontology) and modeled using [Networkx library](https://networkx.github.io/documentation/stable/index.html).

## Link to data

In this [link](https://mega.nz/#F!9LphVIrL!MrxHfvfdHboXxdUoLkmsVg) you can find all the data to reproduce the experiments.
