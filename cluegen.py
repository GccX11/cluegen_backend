import os
import numpy as np
from projects.cluegen_backend.glove_db import Glove
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
from nltk.stem.porter import PorterStemmer


class _Leaf(object):
    def __init__(self, word):
        self.lemma = word
    
    def __str__(self):
        return self.lemma


class _Cluster(object):
    def __init__(self, distance, children):
        self.distance = distance
        self.children = children
        self.lemmas = []
    
    def get_lemmas(self):
        # populate the lemmas array if it is empty
        if len(self.lemmas) == 0:
            for child in self.children:
                if isinstance(child, _Leaf):
                    self.lemmas.append(child.lemma)
                else:
                    self.lemmas.extend(child.get_lemmas())
        return self.lemmas
    
    def __str__(self):
        return str(self.get_lemmas()) + ', ' + str(np.round(self.distance, 2))


class ClueGenerator(object):
    def __init__(self, path):
        # load the glove database
        self.glove_db = Glove(path)

        # words originally from https://github.com/first20hours/google-10000-english
        # with additional filtering by me
        self.all_clue_words = np.load(os.path.join(path, 'all_clue_words.npy'))
        self.all_clue_vectors = np.load(os.path.join(path, 'all_clue_vectors.npy'))

        # TODO: consider precomputing this
        #       (but memory may be too much and this is fast)
        print('stemming clue words')
        self.stemmer = PorterStemmer()
        self.all_stemmed_words = np.array([self.stemmer.stem(word) for word in self.all_clue_words])

        print('ClueGenerator initialized')

    def glove(self, word):
        return self.glove_db.vector(word)
    
    def stem(self, word):
        return self.stemmer.stem(word.lower())

    # create a proper tree structure,
    # where each node has the average distance of its children
    def convert_to_tree(self, pairs, words):
        leaves = {}
        clusters = {}

        for i, row in enumerate(pairs):
            if row[0] < len(words):
                # if it is an original point read it from the centers array
                a = words[int(row[0])]
                a = _Leaf(a)
                leaves[row[0]] = a
            else:
                # other wise read the cluster that has been created
                a = clusters[int(row[0])]

            if row[1] < len(words):
                b = words[int(row[1])]
                b = _Leaf(b)
                leaves[row[1]] = b
            else:
                b = clusters[int(row[1])]

            # set a and b as children of the new node
            distance = row[2]
            cluster = _Cluster(distance, children=[a, b])

            clusters[1 + i + len(pairs)] = cluster
        return clusters

    def cluster_words(self, words, max_cluster_size=5):
        # use the wordA.similarity(wordB) metric
        # to get a similarity matrix
        vectors = np.array([self.glove(word.lower()) for word in words])
        similarity_matrix = vectors @ vectors.T
        vectors_normed_a = np.sqrt(np.sum(vectors**2, axis=1))
        vectors_normed_b = np.sqrt(np.sum(vectors**2, axis=1))
        similarity_matrix = 1 - similarity_matrix / (vectors_normed_a * vectors_normed_b)
        condensed_similarity_matrix = pdist(similarity_matrix)
        # use the similarity matrix to cluster the words
        Z = linkage(condensed_similarity_matrix, 'average')

        clusters = self.convert_to_tree(Z, words)

        # remove clusters that are too big and far apart
        clusters = {k: v for k, v in clusters.items() if len(v.get_lemmas()) <= max_cluster_size}
        #clusters = {k: v for k, v in clusters.items() if v.distance <= max_distance}
        return list(clusters.values())

    def no_word_overlap(self, word, words):
        # make sure the word does not equal,
        # or is not a substring of any of the words, and vice versa
        return (word not in words) and (not any([word in w for w in words]) and (not any([w in word for w in words])))

    def get_clues(self, cluster_words, other_words, num_clues=3):
        cluster_words = set(w.lower() for w in cluster_words)
        other_words = set(w.lower() for w in other_words)
        cluster_stems = set(self.stem(w) for w in cluster_words)
        other_stems = set(self.stem(w) for w in other_words)

        # get the distances between the cluster words and all the clue words
        vectors = np.array([self.glove(word.lower()) for word in cluster_words])
        dists = self.all_clue_vectors @ vectors.T
        vectors_normed = np.sqrt(np.sum(vectors**2, axis=1))
        all_clue_vectors_normed = np.sqrt(np.sum(self.all_clue_vectors**2, axis=1))
        dists = 1 - dists / (np.expand_dims(vectors_normed, 0) * np.expand_dims(all_clue_vectors_normed, 1))
        
        # get the distances between the cluster words and all the other words
        # TODO: this approach doesn't work, but it is still worth considering in future
        #other_vectors = np.array([self.glove(word.lower()) for word in other_words])
        #other_dists = self.all_clue_vectors @ other_vectors.T
        #other_dists = 1 - other_dists / (np.expand_dims(np.linalg.norm(other_vectors, axis=1),0) * np.expand_dims(np.linalg.norm(self.all_clue_vectors, axis=1), 1))
        
        best_dists = np.mean(dists, axis=1) #- np.sum(other_dists, axis=1)
        clue_idxs = np.argsort(best_dists)[:num_clues+len(cluster_words)+len(other_words)+2*num_clues]
        clues = self.all_clue_words[clue_idxs]
        
        # filter out words that are already in the cluster
        clue_idxs = [i for i in range(len(clues)) if self.no_word_overlap(self.stem(clues[i]), cluster_stems) and self.no_word_overlap(self.stem(clues[i]), other_stems)]
        
        # filter out overlapping clues
        words_stems = set()
        filtered_clue_idxs = []
        for clue_idx in clue_idxs:
            word_stem = self.stem(clues[clue_idx])
            if word_stem in words_stems:
                continue
            words_stems.add(word_stem)
            filtered_clue_idxs.append(clue_idx)

        clues = list(clues[filtered_clue_idxs])
        best_dists = list(best_dists[filtered_clue_idxs])
        return clues[:num_clues]

    def generate(self, words, max_cluster_size=5, num_clues=3):
        print('got', len(words), 'words')

        # filter out words that don't have a vector
        words = [word for word in words if self.glove(word.lower()) is not None]
        print('got', len(words), 'words with vectors')
        if len(words) == 0:
            return [], []

        # cluster the words
        clusters = self.cluster_words(words, max_cluster_size=max_cluster_size)
        clusters_words = []
        for cluster in clusters:
            cluster_words = cluster.get_lemmas()
            clusters_words.append(cluster_words)
        print('got', len(clusters_words), 'clusters')

        # get clues for each cluster
        clues_words = []
        for cluster_words in clusters_words:
            other_words = [w for w in words if w not in cluster_words]
            clues_words.append(self.get_clues(cluster_words, other_words, num_clues=num_clues))
        print('got', len(clues_words), 'clues')
        print('----------------')

        return clusters_words, clues_words

