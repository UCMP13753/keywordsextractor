import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import pandas as pd
import seaborn as sns
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span
from collections import defaultdict
import operator
import networkx as nx
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
pd.set_option('display.max_colwidth', 200)
# Get the list of stop words
stop_words = stopwords.words('english')
# add new stopwords to the list
stop_words.extend(["could", "though", "would", "also", "many", 'much', 'may', "water", "de"])
stop_words = set(stop_words)
nlp = spacy.load('en_core_web_sm')


# find the tokens in input_text and count their frequencies
def get_df(input_text):
    list_words = input_text.split(' ')
    set_words_full = list(set(list_words))

    # remove stop words
    set_words = [i for i in set_words_full if i not in stop_words and len(i)>=1]

    # count each word
    count_words = [list_words.count(i) for i in set_words]

    # create DataFrame
    df = pd.DataFrame(zip(set_words, count_words), columns=['words', 'count'])
    df.sort_values('count', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# define a color dictionary for display
def get_colordict(palette, number, start):
    pal = list(sns.color_palette(palette=palette, n_colors=number).as_hex())
    color_d = dict(enumerate(pal, start=start))
    return color_d


# combine the paragraphs in the specified topic and output it in form of string
def create_text(data, topic):
    text_list = []
    if topic == "":
        text_list = data['Paragraph'].to_list()
    else:
        if topic in ['diversity', 'students', 'programs', 'data security', 'legal', 'marketing', 'cost', 'income',
                     'recruiting', 'data management', 'graduation rate']:
            text_list = data[data['topic'] == topic]['Paragraph'].to_list()
        else:
            print("topic invalid")

    text = ' '.join(text_list)
    return text


def remove_symbols(text):
    symbols = "'\<>?;,:%.#@&()—"
    for i in range(len(symbols)):
        text = np.char.replace(text, symbols[i], '')
    return str(text)


def remove_numbers(text):
    text = re.sub(r" \d", "", text)
    return str(text)


def convert_lower_case(text):
    return str(np.char.lower(text))


def remove_stopwords(text):
    tokens = ' '.join([x for x in word_tokenize(convert_lower_case(text)) if x not in stop_words])
    return tokens


# this function can detect the entities pairs in a sentence and out put them in form of list
def get_entities(sent):
    ## chunk 1
    ent1 = ""
    ent2 = ""

    prv_tok_dep = ""  # dependency tag of previous token in the sentence
    prv_tok_text = ""  # previous token in the sentence

    prefix = ""
    modifier = ""
    #############################################################
    for tok in nlp(sent):
        ## chunk 2
        # if token is a punctuation mark then move on to the next token
        print(f"tok: {tok}, type: {tok.dep_}")
        if tok.dep_ != "punct":
            # check: token is a compound word or not
            if tok.dep_ == "compound":
                prefix = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            ## chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

                ## chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            ## chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
    #############################################################

    return [ent1.strip(), ent2.strip()]


# this function can detect the verb which represents the relationship between two entities
def get_relation(sent):
    doc = nlp(sent)
    # Matcher class object
    matcher = Matcher(nlp.vocab)
    #define the pattern
    pattern = [{'DEP':'ROOT'},
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},
            {'POS':'ADJ','OP':"?"}]

    matcher.add("matching_1", [pattern])

    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]]
    return span.text


# we create the graphe G for networkx by specifying the node,
# then we calculate the weight according to co-occurrence matrix
def build_graph_info(text):
    com = defaultdict(lambda: defaultdict(int))
    # data pre-processing
    s = convert_lower_case(remove_numbers(remove_symbols(text)))
    terms_only = [token for token in s.split(" ") if token != "" and token not in stop_words]

    # Build co-occurrence matrix
    for i in range(len(terms_only) - 1):
        for j in range(i + 1, len(terms_only)):
            w1, w2 = sorted([terms_only[i], terms_only[j]])
            if w1 != w2:
                com[w1][w2] += 1

    com_max = []
    # For each term, look for the most common co-occurrent terms
    for t1 in com:
        t1_max_terms = sorted(com[t1].items(), key=operator.itemgetter(1), reverse=True)[:50]
        for t2, t2_count in t1_max_terms:
            com_max.append((t1, t2, t2_count))
    # Get the most frequent co-occurrences
    terms_max = sorted(com_max, key=operator.itemgetter(2), reverse=True)
    df_for_graph = pd.DataFrame(terms_max[:20], columns=['node1', 'node2', 'co_occurence'])
    G = nx.from_pandas_edgelist(df_for_graph, 'node1', 'node2')
    for v in G:
        G.nodes[v]["class"] = G.degree(v)
    weights = list(df_for_graph['co_occurence'])
    return G, weights


# Class available from https://matplotlib.org/3.5.0/gallery/misc/packed_bubbles.html
class BubbleChart:
    def __init__(self, area, bubble_spacing=0):
        area = np.asarray(area)
        r = np.sqrt(area / np.pi)

        self.bubble_spacing = bubble_spacing
        self.bubbles = np.ones((len(area), 4))
        self.bubbles[:, 2] = r
        self.bubbles[:, 3] = area
        self.maxstep = 2 * self.bubbles[:, 2].max() + self.bubble_spacing
        self.step_dist = self.maxstep / 2

        # calculate initial grid layout for bubbles
        length = np.ceil(np.sqrt(len(self.bubbles)))
        grid = np.arange(length) * self.maxstep
        gx, gy = np.meshgrid(grid, grid)
        self.bubbles[:, 0] = gx.flatten()[:len(self.bubbles)]
        self.bubbles[:, 1] = gy.flatten()[:len(self.bubbles)]

        self.com = self.center_of_mass()

    def center_of_mass(self):
        return np.average(
            self.bubbles[:, :2], axis=0, weights=self.bubbles[:, 3]
        )

    def center_distance(self, bubble, bubbles):
        return np.hypot(bubble[0] - bubbles[:, 0],
                        bubble[1] - bubbles[:, 1])

    def outline_distance(self, bubble, bubbles):
        center_distance = self.center_distance(bubble, bubbles)
        return center_distance - bubble[2] - bubbles[:, 2] - self.bubble_spacing

    def check_collisions(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        return len(distance[distance < 0])

    def collides_with(self, bubble, bubbles):
        distance = self.outline_distance(bubble, bubbles)
        idx_min = np.argmin(distance)
        return idx_min if type(idx_min) == np.ndarray else [idx_min]

    def collapse(self, n_iterations=50):

        for _i in range(n_iterations):
            moves = 0
            for i in range(len(self.bubbles)):
                rest_bub = np.delete(self.bubbles, i, 0)
                # try to move directly towards the center of mass
                # direction vector from bubble to the center of mass
                dir_vec = self.com - self.bubbles[i, :2]

                # shorten direction vector to have length of 1
                dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))

                # calculate new bubble position
                new_point = self.bubbles[i, :2] + dir_vec * self.step_dist
                new_bubble = np.append(new_point, self.bubbles[i, 2:4])

                # check whether new bubble collides with other bubbles
                if not self.check_collisions(new_bubble, rest_bub):
                    self.bubbles[i, :] = new_bubble
                    self.com = self.center_of_mass()
                    moves += 1
                else:
                    # try to move around a bubble that you collide with
                    # find colliding bubble
                    for colliding in self.collides_with(new_bubble, rest_bub):
                        # calculate direction vector
                        dir_vec = rest_bub[colliding, :2] - self.bubbles[i, :2]
                        dir_vec = dir_vec / np.sqrt(dir_vec.dot(dir_vec))
                        # calculate orthogonal vector
                        orth = np.array([dir_vec[1], -dir_vec[0]])
                        # test which direction to go
                        new_point1 = (self.bubbles[i, :2] + orth *
                                      self.step_dist)
                        new_point2 = (self.bubbles[i, :2] - orth *
                                      self.step_dist)
                        dist1 = self.center_distance(
                            self.com, np.array([new_point1]))
                        dist2 = self.center_distance(
                            self.com, np.array([new_point2]))
                        new_point = new_point1 if dist1 < dist2 else new_point2
                        new_bubble = np.append(new_point, self.bubbles[i, 2:4])
                        if not self.check_collisions(new_bubble, rest_bub):
                            self.bubbles[i, :] = new_bubble
                            self.com = self.center_of_mass()

            if moves / len(self.bubbles) < 0.1:
                self.step_dist = self.step_dist / 2

    def plot(self, ax, labels, colors):

        for i in range(len(self.bubbles)):
            circ = plt.Circle(
                self.bubbles[i, :2], self.bubbles[i, 2], color=colors[i])
            ax.add_patch(circ)
            ax.text(*self.bubbles[i, :2], labels[i],
                    horizontalalignment='center', verticalalignment='center')


def get_ngrams(text, ngram_from=2, ngram_to=2, n=None, max_features=20000):
    """
    create a reusable function to perform n-gram analysis on a Pandas dataframe column. This will use CountVectorizer
    to create a matrix of token counts found in our text. We’ll use the ngram_range parameter to specify the size of
    n-grams we want to use, so 1, 1 would give us unigrams (one word n-grams) and 1-3, would give us n-grams from one
    to three words.
    https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
    """
    vec = CountVectorizer(ngram_range=(ngram_from, ngram_to),
                          max_features=max_features,
                          stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    if n == None:
        return words_freq
    else:
        return words_freq[:n]


# create df_cont which include the information of top words, their frequencies and their belonging topic.
# we can also define the contents according to our data
def get_multi_topics_df(data):
    contents = ['diversity', 'students', 'programs', 'data security', 'legal', 'marketing', 'cost', 'income',
                'recruiting', 'data management', 'graduation rate']
    pr_text = [create_text(data, content) for content in contents]
    # combine the tokenized words in the same topic
    # clean text in each content
    cn_clean_text = [convert_lower_case(remove_numbers(remove_symbols(t))) for t in pr_text]

    # create DataFrame from top 10 words most appear in each content
    df_cn_words = [list(get_df(i)['words'][0:10]) for i in cn_clean_text]
    df_cn_count = [list(get_df(i)['count'][0:10]) for i in cn_clean_text]
    df_cn_content = [[i.lower()] * len(j) for i, j in zip(contents, df_cn_words)]

    df_cont = pd.DataFrame(zip(sum(df_cn_content, []), sum(df_cn_words, []), sum(df_cn_count, [])),
                           columns=['contents', 'words', 'count'])
    return df_cont, contents





