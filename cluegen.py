import os
import spacy
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


nlp = spacy.load('en_core_web_lg')


class _Leaf(object):
    def __init__(self, word):
        self.lemma = nlp(word)
    
    def __str__(self):
        return self.lemma.text


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
        # words originally from https://github.com/first20hours/google-10000-english
        # with additional filtering by me
        self.all_clue_words = np.load(os.path.join(path, 'all_clue_words.npy'))
        self.all_clue_vectors = np.load(os.path.join(path, 'all_clue_vectors.npy'))

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

    def cluster_words(self, words, max_cluster_size=5, max_distance=1.2):
        words = [nlp(word) for word in words]
        # use the wordA.similarity(wordB) metric
        # to get a similarity matrix
        similarity_matrix = np.zeros((len(words), len(words)))
        for i in range(len(words)):
            for j in range(len(words)):
                similarity_matrix[i, j] = words[i].similarity(words[j])
        condensed_similarity_matrix = pdist(similarity_matrix)
        # use the similarity matrix to cluster the words
        Z = linkage(condensed_similarity_matrix, 'ward')

        clusters = self.convert_to_tree(Z, words)

        # remove clusters that are too big and far apart
        clusters = {k: v for k, v in clusters.items() if len(v.get_lemmas()) <= max_cluster_size}
        clusters = {k: v for k, v in clusters.items() if v.distance <= max_distance}
        return list(clusters.values())

    def no_word_overlap(self, word, words):
        # make sure the word does not equal,
        # or is not a substring of any of the words, and vice versa
        return (word not in words) and (not any([word in w for w in words]) and (not any([w in word for w in words])))

    def get_clue(self, cluster, num_clues=3):
        cluster_words = [lemma.text for lemma in cluster.get_lemmas()]
        cluster_vectors = np.array([lemma.vector for lemma in cluster.get_lemmas()])

        # filter out clues that overlap with the words
        possible_clues = []
        possible_vectors = []
        for clue, vector in zip(self.all_clue_words, self.all_clue_vectors):
            if self.no_word_overlap(clue, cluster_words):
                possible_clues.append(clue)
                possible_vectors.append(vector)
        possible_clues = np.array(possible_clues)
        possible_vectors = np.array(possible_vectors)

        # calculate the cosine distance between each clue and each word
        cluster_dists = possible_vectors @ cluster_vectors.T
        cluster_dists = 1 - cluster_dists / np.expand_dims(np.linalg.norm(cluster_vectors, axis=1),0) * np.expand_dims(np.linalg.norm(possible_vectors, axis=1), 1)
        best_dists = np.sum(cluster_dists, axis=1) # - np.sum(your_dists, axis=1)
        clue_idxs = np.argsort(best_dists)[:num_clues]
        return list(possible_clues[clue_idxs])

    def generate(self, words):
        # cluster the words
        clusters = self.cluster_words(words)

        # get the clue for each cluster
        clusters_words = []
        for cluster in clusters:
            clusters_words.append([w.text for w in cluster.get_lemmas()])
        
        clues_words = []
        for cluster in clusters:
            clues_words.append(self.get_clue(cluster))

        return clusters_words, clues_words

