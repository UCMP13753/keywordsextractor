from keyword_extractor import *
import pandas as pd

from utils import *

# print(get_entities("100% of TD's GHG emissions from electricity 95% of carbon offsets generated social value"))
with open("api_key.txt", "r") as file:
    api_key = file.read()

# instanciate the keywords extractor

ke = KeywordsExtractor(api_key)

###############################################################
# read a csv content and extract keywords from paragraphs of each topic
# in production, instead reading a csv, the content is read from db with columns "Paragraph" and "topic"
# path1 = "datas/reports/test.csv"
# df = pd.read_csv(path1)

# For each topic, we test all the extractors and write the results in different files according to the topic
# for topic in ['data security', 'diversity', 'legal', 'marketing', 'cost', 'income', 'recruiting',
#               'data management', 'graduation rate']:
#     ke.export_google_doc(df, topic)
#
# text = """
# from  Chegg can be accessed anytime and anywhere and are viewed through our eTextbook reader that enables fast and
# easy  navigation, keyword search, text highlighting, note taking and further preserves those notes in an online notepad
# with the ability  to view highlighting and notes across platforms.  Rising Higher Ed tuition has threatened
# affordability and access, leaving many students  with onerous debt or unable to afford college  altogether.
# Education Affordability  Required Materials includes our print textbook and eTextbook offerings, which help students
# save money compared to  the cost of buying new. We offer an extensive print textbook library primarily for rent and
# also for sale both on our own and  through our print textbook partners. We partner with a variety of third parties
# to source print textbooks and eTextbooks directly  or indirectly from publishers.  Demand for trained workers
# continues to  increase but displaced workers need more  affordable and shorter pathways from education  to employment.
# """
file = open("final_clean_text_CleanTxtFiles.txt", encoding='utf8', errors='ignore')
text = file.read()
file.close()
text = remove_symbols(remove_numbers(remove_stopwords(text)))
print(len(text.split()))
print(len(set(text.split())))
#
tfidf = ke.extract_keywords(text, num_keywords=100, method='tf-idf', save_txt=True)["tf-idf"]
print(tfidf)
keybert = ke.extract_keywords(text, num_keywords=100, method='keybert', save_txt=True)["keybert"]
print(keybert)
# print it or save it

###############################################################



##############################################################
#visualize the data in different way

# ke.wordcloud(text, True)

# ke.n_grams_wordcloud(text, 1, 3)

# ke.get_relation_plot(text)

# ke.network_draw_shell(text)

# ke.network_circosplot(text)

# 

ke.multi_bar_chart(text)

# # ke.multi_topics_chart(df)
#
# # ke.sunburst(df)
#
ke.treemap(text)

# # ke.multi_topics_treemap(df)

ke.circle_packing(text)

ke.bubble_chart_plot(text, mode="mix")
# # ke.multi_topics_circle_packing(df)
