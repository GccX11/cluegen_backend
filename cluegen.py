import spacy

nlp = spacy.load('en_core_web_lg')

# load top_words.txt and get the lemma for them in a map
# top words from: https://www.ef.edu/english-resources/english-vocabulary/top-3000-words/
all_clues = []
with open('top_words.txt', 'r') as f:
    for line in f:
        token = line.strip()
        lemma = nlp(token)
        if lemma.has_vector:
            all_clues.append(lemma)

def generate(grid):
    print(nlp.vocab.vectors[nlp.vocab.strings['dog']])
    print(grid)
    return 'test clue'


