import pandas as pd
from keybert import KeyBERT
from rake_nltk import Rake
from yake import yake
import nltk, tqdm
from pke import pke
import string
from pke import compute_document_frequency, compute_lda_model
import glob
from string import punctuation
import os
import openai
import json
from utils import *
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
# import news_graph
# from news_graph.news_graph import NewsMining
# from news_graph.graph_show import GraphShow
import plotly.graph_objects as go
import nxviz as nxviz
from nxviz.plots import CircosPlot
import distinctipy
import plotly.express as px
import circlify
import matplotlib.pyplot as plt
from tqdm import tqdm
from nltk.util import ngrams
import networkx as nx
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from spacy.tokens import Span
import networkx as nx
import bs4
import requests
import spacy
from spacy import displacy
from nltk.util import ngrams
import seaborn as sns
from nltk.corpus import stopwords
from nltk import tokenize
import csv


nltk.download('stopwords')
nltk.download('punkt')
pd.set_option('display.max_colwidth', 200)
# Get the list of stop words
stop_words = stopwords.words('english')
# add new stopwords to the list
stop_words.extend(["could", "though", "would", "also", "many", 'much', 'may'])
stop_words = set(stop_words)


class KeywordsExtractor:
    def __init__(self, api):
        self.my_api = api
        stoplist = list(punctuation)
        input_dir = "datas/reports/"

        documents = []
        for doc in glob.glob(input_dir + '*.csv'):
            documents.append(doc)

        compute_document_frequency(documents,
                                   output_file='datas/models/df.tsv.gz',
                                   language='en',  # language of files
                                   normalization="stemming",  # use porter stemmer
                                   stoplist=stoplist,
                                   n=2)

        compute_lda_model(documents,
                          output_file='datas/models/lda.gzip',
                          n_topics=200,
                          language='en',
                          stoplist=stoplist,
                          normalization='stemming')



    def count_vectorizer_extractor(self, text, k):
        from sklearn.feature_extraction.text import CountVectorizer
        coun_vect = CountVectorizer(stop_words='english', max_features=k, ngram_range=(1,3))
        count_matrix = coun_vect.fit_transform(text.split(','))
        count_array = count_matrix.toarray()
        df = pd.DataFrame(data=count_array,columns = coun_vect.get_feature_names_out())
        res = []
        for column in df.columns:
            res.append(column)
        return res

    def tf_idf_extractor(self, text, k):
        # 1. create a TfIdf extractor.
        extractor = pke.unsupervised.TfIdf()

        # 2. load the content of the document.
        stoplist = list(string.punctuation)
        stoplist += pke.lang.stopwords.get('en')
        extractor.load_document(text,
                                language='en',
                                stoplist=stoplist,
                                normalization=None)

        # 3. select {1-3}-grams not containing punctuation marks as candidates.
        extractor.candidate_selection(n=3)

        # 4. weight the candidates using a `tf` x `idf`
        df = pke.load_document_frequency_file(input_file='datas/models/df.tsv.gz')
        extractor.candidate_weighting(df=df)

        # 5. get the k-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    def kp_miner_extractor(self, text, k):
        # 1. create a KPMiner extractor.
        extractor = pke.unsupervised.KPMiner()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en',
                                normalization=None)

        # 3. select {1-2}-grams that do not contain punctuation marks or
        #    stopwords as keyphrase candidates. Set the least allowable seen
        #    frequency to 5 and the number of words after which candidates are
        #    filtered out to 1000.
        lasf = 2
        cutoff = 1000
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

        # 4. weight the candidates using KPMiner weighting function.
        df = pke.load_document_frequency_file(input_file='datas/models/df.tsv.gz')
        alpha = 2.3
        sigma = 3.0
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)

        # 5. get the k-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    def keybert_extractor(self, text, k):
        """
        Uses KeyBERT to extract the top k keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        bert = KeyBERT()
        keywords = bert.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=k)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    def rake_extractor(self, text, k):
        """
        Uses Rake to extract the top k keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        r = Rake()
        r.extract_keywords_from_text(text)    # algorithm cannot handle specification of length of keywords for now . See https://github.com/csurfer/rake-nltk/issues/5
        return r.get_ranked_phrases()[:k]

    # YAKE
    def yake_extractor(self, text, k):
        """
        Uses YAKE to extract the top 5 keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        keywords = yake.KeywordExtractor(lan="en", n=2, windowsSize=10, top=k).extract_keywords(text)
        results = []
        for scored_keywords in keywords:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # TextRank
    def text_rank_extractor(self, text, k):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # 1. create a TextRank extractor.
        extractor = pke.unsupervised.TextRank()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en',
                                normalization=None)

        # 3. build the graph representation of the document and rank the words.
        #    Keyphrase candidates are composed from the 33-percent
        #    highest-ranked words.
        extractor.candidate_weighting(window=2,
                                      pos=pos,
                                      top_percent=0.33)

        # 4. get the k-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)

        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    def position_rank_extractor(self, text, k):
        """
        Uses PositionRank to extract the top k keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        # define the valid Part-of-Speeches to occur in the graph
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(text, language='en', normalization='stemming')
        extractor.candidate_selection(maximum_word_number=2)
        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk biaised with the position of the words
        #    in the document. In the graph, nodes are words (nouns and
        #    adjectives only) that are connected if they occur in a window of
        #    3 words.
        extractor.candidate_weighting(window=10, pos=pos)
        # 5. get the k-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # MultipartiteRank
    def multipartite_rank_extractor(self, text, k):
        """
        Uses MultipartiteRank to extract the top k keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(text, language='en')
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor.candidate_selection(pos=pos)
        # 4. build the Multipartite graph and rank candidates using random walk,
        #    alpha controls the weight adjustment mechanism, see TopicRank for
        #    threshold/method parameters.
        extractor.candidate_weighting(alpha=1.1, threshold=0.74, method='average')
        keyphrases = extractor.get_n_best(n=k)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # TopicRank
    def topic_rank_extractor(self, text, k):
        """
        Uses TopicRank to extract the top k keywords from a text
        Arguments: text (str)
        Returns: list of keywords (list)
        """
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(text, language='en')
        pos = {'NOUN', 'PROPN', 'ADJ', 'ADV'}
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting()
        keyphrases = extractor.get_n_best(n=k)
        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # TopicalPageRank
    def topical_page_rank_extractor(self, text, k):
        # define the valid Part-of-Speeches to occur in the graph
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # define the grammar for selecting the keyphrase candidates
        grammar = "NP: {<ADJ>*<NOUN|PROPN>+}"

        # 1. create a TopicalPageRank extractor.
        extractor = pke.unsupervised.TopicalPageRank()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en',
                                stoplist=None,
                                normalization='stemming')

        # 3. select the noun phrases as keyphrase candidates.
        extractor.candidate_selection(grammar=grammar)

        # 4. weight the keyphrase candidates using Single Topical PageRank.
        #    Builds a word-graph in which edges connecting two words occurring
        #    in a window are weighted by co-occurrence counts.
        extractor.candidate_weighting(window=10,
                                      pos=pos,
                                      lda_model='datas/models/lda.gzip')

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)

        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    # SingleRank
    def single_rank_extractor(self, text, k):
        # define the set of valid Part-of-Speeches
        pos = {'NOUN', 'PROPN', 'ADJ'}

        # 1. create a SingleRank extractor.
        extractor = pke.unsupervised.SingleRank()

        # 2. load the content of the document.
        extractor.load_document(input=text,
                                language='en',
                                normalization=None)

        # 3. select the longest sequences of nouns and adjectives as candidates.
        extractor.candidate_selection(pos=pos)

        # 4. weight the candidates using the sum of their word's scores that are
        #    computed using random walk. In the graph, nodes are words of
        #    certain part-of-speech (nouns and adjectives) that are connected if
        #    they occur in a window of 10 words.
        extractor.candidate_weighting(window=10,
                                      pos=pos)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=k)

        results = []
        for scored_keywords in keyphrases:
            for keyword in scored_keywords:
                if isinstance(keyword, str):
                    results.append(keyword)
        return results

    def find_GPT_keywords(self, api_key, text):
        openai.api_key = api_key
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Extract keywords from this text: \n {text[:4000]} ",
            temperature=0.3,
            max_tokens=120,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.0
        )

        return response['choices'][0]['text']

    """
    we specify the data and the topic we want, and export_google_doc will use 
    the methods above to extract the keywords and write the results in different rows
    if we define the variable my_api, we activate the extractor "find_GPT_keywords";
    otherwise we only test the other extractors.
    """
    def export_google_doc(self, data, topic, with_openai=False):
        text = create_text(data, topic)
        if with_openai == False:
            res = {
                'text': text,
                'tf-idf': ", ".join(self.tf_idf_extractor(text, 50)),
                'kp-miner': ", ".join(self.kp_miner_extractor(text, 50)),
                'rake': ", ".join(self.rake_extractor(text, 50)),
                'yake': ", ".join(self.yake_extractor(text, 50)),
                'text-rank': ", ".join(self.text_rank_extractor(text, 50)),
                'single-rank': ", ".join(self.single_rank_extractor(text, 50)),
                'topic-rank': ", ".join(self.topic_rank_extractor(text, 50)),
                'multipartite-rank': ", ".join(self.multipartite_rank_extractor(text, 50)),
                'keybert': ", ".join(self.keybert_extractor(text, 50)),
                'count-vectorizer': ", ".join(self.count_vectorizer_extractor(text, 50)),
            }
        else:
            res = {
                'text': text,
                'tf-idf': ", ".join(self.tf_idf_extractor(text, 50)),
                'kp-miner': ", ".join(self.kp_miner_extractor(text, 50)),
                'rake': ", ".join(self.rake_extractor(text, 50)),
                'yake': ", ".join(self.yake_extractor(text, 50)),
                'text-rank': ", ".join(self.text_rank_extractor(text, 50)),
                'single-rank': ", ".join(self.single_rank_extractor(text, 50)),
                'topic-rank': ", ".join(self.topic_rank_extractor(text, 50)),
                'multipartite-rank': ", ".join(self.multipartite_rank_extractor(text, 50)),
                'keybert': ", ".join(self.keybert_extractor(text, 50)),
                'count-vectorizer': ", ".join(self.count_vectorizer_extractor(text, 50)),
                'openai': self.find_GPT_keywords(self.my_api, text)
            }

        with open('datas/keywords/keywords-' + topic + '.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in res.items():
                writer.writerow([key, value])

    # if we set method = "all" return a dctionary res, if we specify the method, return a string include all the keywords
    def extract_keywords(self, text, num_keywords=50, method="all", save_txt=False):
        res = {}
        if method == "all":
            res = {
                'tf-idf': ", ".join(self.tf_idf_extractor(text, num_keywords)),
                'kp-miner': ", ".join(self.kp_miner_extractor(text, num_keywords)),
                'rake': ", ".join(self.rake_extractor(text, num_keywords)),
                'yake': ", ".join(self.yake_extractor(text, num_keywords)),
                'text-rank': ", ".join(self.text_rank_extractor(text, num_keywords)),
                'single-rank': ", ".join(self.single_rank_extractor(text, num_keywords)),
                'topic-rank': ", ".join(self.topic_rank_extractor(text, num_keywords)),
                'multipartite-rank': ", ".join(self.multipartite_rank_extractor(text, num_keywords)),
                'keybert': ", ".join(self.keybert_extractor(text, num_keywords)),
                'count-vectorizer': ", ".join(self.count_vectorizer_extractor(text, num_keywords)),
                'openai': self.find_GPT_keywords(self.my_api, text)
            }
        elif method == "tf-idf":
            res = {'tf-idf': ", ".join(self.tf_idf_extractor(text, num_keywords))}
        elif method == "kp-miner":
            res = {'kp-miner': ", ".join(self.kp_miner_extractor(text, num_keywords))}
        elif method == "rake":
            res = {'rake': ", ".join(self.rake_extractor(text, num_keywords)),}
        elif method == "yake":
            res = {'yake': ", ".join(self.yake_extractor(text, num_keywords))}
        elif method == "text-rank":
            res = {'text-rank': ", ".join(self.text_rank_extractor(text, num_keywords))}
        elif method == "single-rank":
            res = {'single-rank': ", ".join(self.single_rank_extractor(text, num_keywords))}
        elif method == "topic-rank":
            res = {'topic-rank': ", ".join(self.topic_rank_extractor(text, num_keywords))}
        elif method == "multipartite-rank":
            res = {'multipartite-rank': ", ".join(self.multipartite_rank_extractor(text, num_keywords))}
        elif method == "keybert":
            res = {'keybert': ", ".join(self.keybert_extractor(text, num_keywords))}
        elif method == "count-vectorizer":
            res = {'count-vectorizer': ", ".join(self.count_vectorizer_extractor(text, num_keywords))}
        elif method == "openai":
            res = {'openai': self.find_GPT_keywords(self.my_api, text)}
        if save_txt:
            with open(f'datas/keywords/extracted-keywords_{method}.txt', 'w') as f:
                for key, value in res.items():
                    f.writelines(key + " :" + "\n")
                    f.writelines(value + "\n")
                    f.writelines("\n")
        return res
        # with open('datas/keywords/extract-keywords-with-' + method + '-method.csv', 'w') as csv_file:
        #     writer = csv.writer(csv_file)
        #     for key, value in res.items():
        #         writer.writerow([key, value])

    # display bigrams in generated wordcloud objects
    def wordcloud(self, text, delete_stopwords=False):
        if delete_stopwords:
            text = remove_stopwords(text)
        wordcloud_text = WordCloud(collocation_threshold=2,
                                   collocations=True,
                                   background_color="white",
                                   colormap='binary', width=1600, height=800).generate(text)
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud_text, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    # display n-grams in generated wordcloud objects, we can specify the number of n
    def n_grams_wordcloud(self, text, ngram_from=2, ngram_to=4, delete_stopwords=False):
        if delete_stopwords:
            text = remove_stopwords(text)
        # sentence tokenize the text because CountVectorizer expect an iterable over raw text documents
        list_text = tokenize.sent_tokenize(text)

        # previous code if we only need one single n for n-gram
        # ngrm = list(ngrams(text.split(), n))
        # map_ngrm = list(map(' '.join, ngrm))
        # counter = Counter(map_ngrm)

        # get the n-gram and create the world cloud
        words_freq = get_ngrams(list_text, ngram_from, ngram_to)
        counter = dict(words_freq)
        wordcloud_trigrams_neg = WordCloud(background_color="white",
                                           colormap='binary', width=1600, height=800).generate_from_frequencies(counter)
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud_trigrams_neg, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    # create the graph of relation based on the library news_graph
    # https://github.com/BrambleXu/news-graph
    # From the previous author's tests, we may need to put all the news_graph code
    # in the current location to get the code to run
    def vis_graph(self, content):
        Miner = NewsMining()
        Miner.main(content)

    # When relation="", we output the whole  relation plot;
    # If we specify a relationship, we output the subject and object belonging to that relationship
    def get_relation_plot(self, text, relation=""):
        # pre-process the data
        list_text = tokenize.sent_tokenize(text)
        # get all the relations in the data
        print(list_text)
        relations = [get_relation(i) for i in tqdm(list_text)]
        # in case we don't know what relationships we have, print them out
        print(relations)
        entity_pairs = []
        for i in tqdm(list_text):
            entity_pairs.append(get_entities(i))
        # extract subject
        source = [i[0] for i in entity_pairs]
        # extract object
        target = [i[1] for i in entity_pairs]
        kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
        # create a directed-graph from a dataframe
        if relation == "":
            G = nx.from_pandas_edgelist(kg_df, "source", "target",
                                    edge_attr=True, create_using=nx.MultiDiGraph())
        else:
            G = nx.from_pandas_edgelist(kg_df[kg_df['edge'] == relation], "source", "target",
                                    edge_attr=True, create_using=nx.MultiDiGraph())
        plt.figure(figsize=(12, 12))
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
        plt.show()

    # Draw the shell diagram that connects all related entities.(Line segments are not weighted)
    def network_draw_shell(self, text):
        G, weights = build_graph_info(text)
        figure(figsize=(10, 7))
        nx.draw_shell(G, with_labels=True)
        plt.show()

    # Draw the circosplot that connects all related entities.(Line segments are weighted)
    def network_circosplot(self, text):
        G, weights = build_graph_info(text)
        c = CircosPlot(G, figsize=(10, 10),
                       node_labels=True,
                       edge_width=weights,
                       node_grouping="class",
                       node_color="class")
        c.draw()
        plt.show()

    # A packed bubble chart displays data in a cluster of circles.
    # create the packed bubble chart based on the frequencies or the keybert score of the top words
    def bubble_chart_plot(self, text, mode="mix"):
        if mode not in ["mix", "frequency", "keybert"]:
            print("mode invalid")
            pass
        else:
            # pre-process the text
            text_list = tokenize.sent_tokenize(text)
            text_list = [remove_symbols(remove_numbers(remove_stopwords(t))) for t in text_list]

            text = remove_symbols(remove_numbers(remove_stopwords(text)))

            # create the dataframe of top words and their frequencies/score, according to the selected mode
            if mode == "mix":
                # get the top 30 words based on their keybert scores
                bert = KeyBERT()
                keywords = bert.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=50)
                # create the dataframe of top words and their keybert score
                data_bubble = pd.DataFrame(keywords, columns=['word', 'score'])
                # set these found n-grams as our vocabulary
                my_vocabulary = data_bubble['word']
                # accept unigrams and bigrams, count their frequencies
                vectorizer = CountVectorizer(ngram_range=(1, 2))
                vectorizer.fit_transform(my_vocabulary)
                term_count = vectorizer.transform(text_list)
                sum_words = term_count.sum(axis=0)
                words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
                words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
                data_bubble = pd.DataFrame(words_freq[:50], columns=['word', 'frequency'])
            elif mode == "frequency":
                # get the top 10 words based on their frequencies
                data_bubble = pd.DataFrame(get_ngrams(text_list, ngram_from=1, ngram_to=2, n=10), columns=['word', 'frequency'])
            else:
                # get the top 10 words based on their keybert scores
                bert = KeyBERT()
                keywords = bert.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=10)

                # create the dataframe of top words and their keybert score
                data_bubble = pd.DataFrame(keywords, columns=['word', 'frequency'])
                # increase the difference between the score for visualization
                min_score = data_bubble['frequency'].min()
                data_bubble['frequency'] = data_bubble['frequency'].apply(lambda x: (x - min_score) * 20 + min_score)
            # define the color for different words
            data_bubble["color"] = distinctipy.get_colors(data_bubble.shape[0])
            data = {
                'words': data_bubble['word'],

                'frequency': data_bubble['frequency'],

                'color': data_bubble['color']
            }
            bubble_chart = BubbleChart(area=data['frequency'],
                                       bubble_spacing=0.1)
            bubble_chart.collapse()
            fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
            fig.set_size_inches(9, 13, forward=True)
            bubble_chart.plot(
                ax, data['words'], data['color'])
            ax.axis("off")
            ax.relim()
            ax.autoscale_view()
            # plt.show()
            plt.savefig('bubble_chart_plot.png')

    # create multiple bar charts and combining them to save space.
    def multi_bar_chart(self, text):
        
        df_words = get_df(text)
        if len(df_words) <= 100:
            print("text too short")
            return
        # separate data into 5 columns
        index_list = [[i[0], i[-1] + 1] for i in np.array_split(range(100), 5)]
        # set the color dictionary according to the maximum of the frequency
        n = df_words['count'].max()
        color_dict = get_colordict('viridis', n, 1)

        fig, axs = plt.subplots(1, 5, figsize=(16, 8), facecolor='white', squeeze=False)
        for col, idx in zip(range(0, 5), index_list):
            df = df_words[idx[0]:idx[-1]]
            # name the label in form of "words: frequency",and set their color according to the frequency
            label = [w + ': ' + str(n) for w, n in zip(df['words'], df['count'])]
            color_l = [color_dict.get(i) for i in df['count']]
            x = list(df['count'])
            y = list(range(0, 20))

            sns.barplot(x=x, y=y, data=df, alpha=0.9, orient='h',
                        ax=axs[0][col], palette=color_l)
            axs[0][col].set_xlim(0, n + 1)  # set X axis range max
            axs[0][col].set_yticklabels(label, fontsize=12)
            axs[0][col].spines['bottom'].set_color('white')
            axs[0][col].spines['right'].set_color('white')
            axs[0][col].spines['top'].set_color('white')
            axs[0][col].spines['left'].set_color('white')

        plt.tight_layout()
        # plt.show()
        plt.savefig('multi_bar_chart.png')

    #  create the Grid of bar charts containing the top 10 most frequent words from each content of the data
    def multi_topics_chart(self, data):
        df_cont, contents = get_multi_topics_df(data)
        # color dictionary
        n = df_cont['count'].max()
        color_dict = get_colordict('viridis', n, 1)

        # create a list contains DataFrame of each content
        keep_dfcon = [df_cont[df_cont['contents'] == i.lower()] for i in contents[0:-1]]
        num_w = len(keep_dfcon)

        fig, axs = plt.subplots(1, num_w, figsize=(16, 6), facecolor='white', squeeze=False)
        for col, df in zip(range(0, num_w), keep_dfcon):
            label = [w + ':' + str(n) for w, n in zip(df['words'], df['count'])]
            color_l = [color_dict.get(i) for i in df['count']]
            x = list(df['count'])
            y = list(range(0, 10))

            sns.barplot(x=x, y=y, data=df, alpha=0.9, orient='h',
                        ax=axs[0][col], palette=color_l)
            axs[0][col].set_xlim(0, n + 1)  # set X axis range max
            axs[0][col].set_yticklabels(label)
            axs[0][col].spines['bottom'].set_color('white')
            axs[0][col].spines['right'].set_color('white')
            axs[0][col].spines['top'].set_color('white')
            axs[0][col].spines['left'].set_color('white')
            title = df['contents'].iloc[0].replace(' ', '\n')
            axs[0][col].set_title(title, y=-0.39)

        plt.tight_layout()
        plt.show()

    # Draw the Sunburst graph with hierarchy, visualize the keywords in each topic
    def sunburst(self, data):
        df_cont, contents = get_multi_topics_df(data)
        df_sum = df_cont.groupby(['contents']).sum().reset_index()
        # create list for pre_words for using with plolty Sunburst Plot
        pre_words = [i.split(' ')[0] for i in list(df_cont['contents'])]
        # create list for keywords for Sunburst Plot, display their frequencies directly after the words
        sb_words = [i + ' ' + str(j) for i, j in zip(list(df_cont["words"]), list(df_cont["count"]))] + list(df_sum['contents'])
        # create list for frequencies for Sunburst Plot to sort the words
        sb_count = list(df_cont["count"]) + list(df_sum['count'])
        # create a list of topics corresponding to the keywords, here we set the main topics as 'ESG'
        sb_contents = list(df_cont["contents"]) + ['ESG'] * len(list(df_sum['contents']))
        list_cn_count = list(df_cont["count"])
        # calculate the difference between the maximum and minimum values for color setting
        nc = max(list_cn_count) - min(list_cn_count) + 1
        color_w = get_colordict('Reds', nc, min(list_cn_count))
        # color dict for contents
        list_sum_count = list(df_sum['count'])
        nw = max(list_sum_count) - min(list_sum_count) + 1
        color_c = get_colordict('Reds', nw, min(list_sum_count))
        # create color list
        sb_color = [color_w.get(i) for i in df_cont["count"]] + [color_c.get(i) for i in list(df_sum['count'])]
        fig = go.Figure(go.Sunburst(labels=sb_words,
                                    parents=sb_contents,
                                    values=sb_count,
                                    marker=dict(colors=sb_color)
                                    ))
        fig.update_layout(width=800, height=800,
                          margin=dict(t=0, l=0, r=0, b=0))
        fig.show()

    # create a treemap with the top 100 words of the selected topic
    def treemap(self, text):
        df_words = get_df(text)
        if len(df_words) <= 100:
            print("text too short")
            return
        fig = px.treemap(df_words[0:100], path=[px.Constant("Top 100 words of text: "), 'words'],
                         values='count',
                         color='count',
                         color_continuous_scale='viridis',
                         color_continuous_midpoint=np.average(df_words['count'])
                         )
        fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
        fig.show()

    # create a treemap with hierarchies.
    # The first level is the topics and the second level is the top 10 words of each topic.
    def multi_topics_treemap(self, data):
        df_cont, contents = get_multi_topics_df(data)
        fig = px.treemap(df_cont, path=[px.Constant("ESG Report"), 'contents', 'words'],
                         values='count',
                         color='count', hover_data=['count'],
                         color_continuous_scale='viridis',
                         color_continuous_midpoint=np.average(df_cont['count'])
                         )
        fig.update_layout(margin=dict(t=0, l=0, r=0, b=0))
        #fig.show()
        fig.write_html("treemap.html")

    # circle packing is a bubble plot with no overlapping area
    # we start with no hierarchy
    def circle_packing(self, text):
        df_words = get_df(text)
        # compute circle positions:
        circles = circlify.circlify(df_words['count'][0:30].tolist(),
                                    show_enclosure=False,
                                    target_enclosure=circlify.Circle(x=0, y=0)
                                    )
        n = df_words['count'][0:30].max()
        color_dict = get_colordict('RdYlBu_r', n, 1)
        fig, ax = plt.subplots(figsize=(9, 9), facecolor='white')
        ax.axis('off')
        lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r, ) for circle in circles)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # list of labels
        labels = list(df_words['words'][0:30])
        counts = list(df_words['count'][0:30])
        labels.reverse()
        counts.reverse()

        # print circles
        for circle, label, count in zip(circles, labels, counts):
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, alpha=0.9, color=color_dict.get(count)))
            plt.annotate(label + '\n' + str(count), (x, y), size=12, va='center', ha='center')
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig('circle_packing.png')

    # circle packing with hierarchy
    def multi_topics_circle_packing(self, data):
        df_cont, contents = get_multi_topics_df(data)
        df_cont['topic'] = ['ESG Report'] * len(df_cont)
        df_cont = df_cont[['topic', 'contents', 'words', 'count']]

        # adjust data format ('id', 'datum', 'children') for working with circlify
        keep_sub = []
        for ii in list(set(df_cont.iloc[:, 1])):
            df_lv2 = df_cont[df_cont[df_cont.columns[1]] == ii]
            df_lv2_gb = df_cont.groupby([df_cont.columns[1]]).sum().reset_index()
            df_lv2_gbii = df_lv2_gb[df_lv2_gb['contents'] == ii]

            se_lv3 = [{'id': i, 'datum': j} for i, j in zip(df_lv2['words'], df_lv2['count'])]
            dict_lv2_lv3 = {'id': df_lv2_gbii.iloc[0, 0], 'datum': df_lv2_gbii.iloc[0, 1], 'children': se_lv3}
            keep_sub.append(dict_lv2_lv3)

        df_lv1_gb = df_cont.groupby([df_cont.columns[0]]).sum().reset_index()
        data = [{'id': df_lv1_gb.iloc[0, 0], 'datum': df_lv1_gb.iloc[0, 1], 'children': keep_sub}]
        circles = circlify.circlify(data,
                                    show_enclosure=False,
                                    target_enclosure=circlify.Circle(x=0, y=0, r=0.5)
                                    )
        n = df_cont['count'].max()
        color_dict_lv3 = get_colordict('RdYlBu_r', n, 1)
        fig, ax = plt.subplots(figsize=(18, 18), facecolor='white')
        ax.axis('off')
        # Find axis boundaries
        lim = max(max(abs(circle.x) + circle.r, abs(circle.y) + circle.r, ) for circle in circles)
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)

        # Print circle the highest level (contents):
        for circle in circles:
            if circle.level != 2:
                continue
            x, y, r = circle
            ax.add_patch(plt.Circle((x, y), r, color="white"))

        # Print circle and labels for the highest level:
        for circle in circles:
            if circle.level != 3:
                continue
            x, y, r = circle
            label = circle.ex["id"]
            ax.add_patch(plt.Circle((x, y), r, color=color_dict_lv3.get(circle.ex['datum'])))
            # annotate each circle
            plt.annotate(label + '\n' + str(circle.ex['datum']), (x, y), ha='center', color="black")

        # Print labels for the contents
        for circle in circles:
            if circle.level != 2:
                continue
            x, y, r = circle
            label = circle.ex["id"]
            plt.annotate(label, (x, y), va='center', ha='center', size=12,
                         bbox=dict(facecolor='white', edgecolor='white',
                                   alpha=0.6, boxstyle='round', pad=0.2)
                         )
        plt.show()
