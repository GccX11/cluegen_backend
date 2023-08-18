import spacy
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


nlp = spacy.load('en_core_web_lg')

# load top_words.txt and get the lemma for them in a map
# top words from: https://www.ef.edu/english-resources/english-vocabulary/top-3000-words/
all_clues = []
with open('projects/cluegen_backend/top_words.txt', 'r') as f:
    for line in f:
        token = line.strip()
        lemma = nlp(token)
        if lemma.has_vector:
            all_clues.append(lemma)


class Leaf(object):
    def __init__(self, word):
        self.lemma = nlp(word)
    
    def __str__(self):
        return self.lemma.text


class Cluster(object):
    def __init__(self, distance, children):
        self.distance = distance
        self.children = children
        self.lemmas = []
    
    def get_lemmas(self):
        # populate the lemmas array if it is empty
        if len(self.lemmas) == 0:
            for child in self.children:
                if isinstance(child, Leaf):
                    self.lemmas.append(child.lemma)
                else:
                    self.lemmas.extend(child.get_lemmas())
        return self.lemmas
    
    def __str__(self):
        return str(self.get_lemmas()) + ', ' + str(np.round(self.distance, 2))


# create a proper tree structure,
# where each node has the average distance of its children
def convert_to_tree(pairs, words):
    leaves = {}
    clusters = {}

    for i, row in enumerate(pairs):
        if row[0] < len(words):
            # if it is an original point read it from the centers array
            a = words[int(row[0])]
            a = Leaf(a)
            leaves[row[0]] = a
        else:
            # other wise read the cluster that has been created
            a = clusters[int(row[0])]

        if row[1] < len(words):
            b = words[int(row[1])]
            b = Leaf(b)
            leaves[row[1]] = b
        else:
            b = clusters[int(row[1])]

        # set a and b as children of the new node
        distance = row[2]
        cluster = Cluster(distance, children=[a, b])

        clusters[1 + i + len(pairs)] = cluster
    return clusters


def cluster_words(words, max_cluster_size=5):
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

    clusters = convert_to_tree(Z, words)

    # remove clusters that are too big
    clusters = {k: v for k, v in clusters.items() if len(v.get_lemmas()) <= max_cluster_size}

    return clusters.values()


def get_clue(cluster):
    all_clues_less_cluster = [w for w in all_clues if w.text.lower() not in [c.text.lower() for c in cluster.get_lemmas()]]

    # get the average vector for the cluster
    cluster_avg = np.mean([w.vector for w in cluster.get_lemmas()], axis=0)
    # get the distance from each word to the cluster average
    distances = [np.linalg.norm(w.vector - cluster_avg) for w in all_clues_less_cluster]
    # get the index of the word with the smallest distance
    clue_index = np.argmin(distances)
    # return the word with the smallest distance
    clue = all_clues_less_cluster[clue_index]
    return clue


def generate(grid):

    # get the words from the grid
    words = []
    for row in grid:
        for cell in row:
            revealed = cell['revealed']
            # only consider unrevealed words
            if not revealed:
                words.append(cell['word'])
    
    # cluster the words
    clusters = cluster_words(words)

    # get the clue for each cluster
    clusters_words = []
    for cluster in clusters:
        clusters_words.append([w.text for w in cluster.get_lemmas()])
    
    clues_words = []
    for cluster in clusters:
        clues_words.append(get_clue(cluster).text)

    return clusters_words, clues_words

