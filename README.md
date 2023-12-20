This library reads the text and extract the keywords using different extractor. We can manually select the method we want to extract keywords. 


---

## Installation
```
python -m pip install -r requirements.txt
```

---

## How to Test

To test the keywrods extractor, create a file named ```api_key.txt``` and put your open ai API key for GPT in it. 
If you don't want to use GPT, you can create an empty api_key.txt
- Rename your desired data csv file to test.csv and put in the data folder. Then run the following:
```
python test.py
```

## How to Use
### Extract keywords
To extract the keywords from a text, instanciate the keywrods extractor
```
ke = KeywordsExtractor(api_key)
```


Extract the keywords from a single text block, we can define the number of keywords we want, the method we use and if we want to save this result as a txt file.

The method list is: ['all', 'tf-idf', 'kp-miner', 'rake', 'yake', 'text-rank', 'single-rank', 'topic-rank', 'multipartite-rank', 'keybert', 'count-vectorizer', 'openai']

The out put will be in forme of dictionary, keys are the using methods, values are the extracted keywords of this method.
```
res = ke.extract_keywords(text, num_keywords=10, method='keybert', save_txt=True)
```


the default parameters: extract_keywords(self, text, num_keywords=50, method="all", save_txt=False)
```
res = ke.extract_keywords(text)
```
### Visualization
There are many built-in visualization functions in KeywordsExtractor. Most of them accept single text block as the input.


We can select if we want to remove the stopword when display the wordcloud.

the default parameters: wordcloud(self, text, delete_stopwords=False)
```
ke.wordcloud(text, delete_stopwords=True)
```


We can select the ngram range by modifying the value of ngram_from and ngram_to.

the default parameters: n_grams_wordcloud(self, text, ngram_from=2, ngram_to=4, delete_stopwords=False)
```
ke.n_grams_wordcloud(text, 1, 3)
```


When relation="", we output the whole relation plot; if we specify a relationship, we output the subject and object belonging to that relationship.

the default parameters: get_relation_plot(self, text, relation="")
```
ke.get_relation_plot(text, relation="accessed")
```


Draw the shell diagram that connects all related entities. (Line segments are not weighted)
```
ke.network_draw_shell(text)
```


Draw the circosplot that connects all related entities. (Line segments are weighted)
```
ke.network_circosplot(text)
```


Create the packed bubble chart based on the frequencies or the keybert score of the top words.

the default parameters:bubble_chart_plot(self, text, mode="mix")
```
ke.bubble_chart_plot(text, mode="mix")
```


Create multiple bar charts and combining them to save space. 

(It will have error if the detected top wrods are less than 100,make sure you have a long text tovisualize)
```
ke.multi_bar_chart(text)
```


Create a treemap with the top 100 words of the selected topic.

(It will have error if the detected top wrods are less than 100,make sure you have a long text tovisualize)
```
ke.treemap(text)
```


Circle packing is a bubble plot with no overlapping area.
```
ke.circle_packing(text)
```


There are also some more complicate visualization which can show the relationship with hierarchy.

Meanwhile, their input need to be a dataframe with columns 'Paragraph' and 'topic'.
```
ke.multi_topics_chart(df)

ke.sunburst(df)

ke.multi_topics_treemap(df)

ke.multi_topics_circle_packing(df)
```

---

