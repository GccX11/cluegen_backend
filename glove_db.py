import os
import numpy as np
import sqlite3

class Glove(object):
    def __init__(self, path):
        self.path = path

    def vector(self, word):
        with sqlite3.connect(os.path.join(self.path, 'glove.db')) as con:
            cur = con.cursor()
            vector_str = cur.execute("SELECT vector FROM word WHERE string=?", (word,)).fetchone()[0]
            return np.array(vector_str.split(','), dtype=np.float32)



# setup functions, disabled for now
#import tqdm
# def load_glove_vectors(path):
#     glove_dict = {}
#     with open(path, 'r') as f:
#         for line in f:
#             line = line.strip().split()
#             glove_dict[line[0]] = np.array(line[1:], dtype=np.float32)
#     return glove_dict
# glove_mem = load_glove_vectors('glove.6B.300d.txt')
# def populate_word():
#     for word in tqdm.tqdm(glove_mem):
#         with sqlite3.connect('glove.db') as con:
#             cur = con.cursor()
#             cur.execute("INSERT INTO word(string, vector) VALUES(?,?)",
#                         (word, ','.join(glove_mem[word].astype(str))))
#             con.commit()
# def validate():
#     """
#     Validate that the glove vectors in the DB are the same as the ones in memory.
#     """
#     glove_mem = load_glove_vectors('glove.6B.300d.txt')
#     import glove_db as glove
#     same = []
#     for word in tqdm.tqdm(glove_mem):
#         same.append(np.all(glove.vector(word) == glove_mem[word]))
#     return np.all(same)

