import os
import numpy as np
import sqlite3

class Glove(object):
    def __init__(self, path):
        self.path = path

    def vector(self, word):
        with sqlite3.connect(os.path.join(self.path, 'glove.db')) as con:
            cur = con.cursor()
            vector_str = cur.execute("SELECT vector FROM word WHERE word="+"'"+word+"';").fetchone()[0]
            return np.array(vector_str.split(','), dtype=np.float32)


# setup functions, disable for now
#import tqdm

# def initialize():
#     db.connect()
#     db.create_tables([Word], safe = True)
#     db.close()

# def load_glove_vectors(path):
#     glove_dict = {}
#     with open(path, 'r') as f:
#         for line in f:
#             line = line.strip().split()
#             glove_dict[line[0]] = np.array(line[1:], dtype=np.float32)
#     return glove_dict

# def populate_db():
#     glove = load_glove_vectors('glove.6B.300d.txt')
#     for word in tqdm.tqdm(glove):
#         word_db = Word(word=str(word), vector=','.join(glove['dog'].astype(str)))
#         word_db.save()

# def validate():
#     """
#     Validate that the glove vectors in the DB are the same as the ones in memory.
#     """
#     glove_mem = load_glove_vectors('glove.6B.300d.txt')
#     import glove_db as glove
#     same = []
#     for word in tqdm.tqdm(glove_mem):
#         same.append(np.all(glove.vector('dog') == glove_mem['dog']))
#     return np.all(same)

