import nltk
#nltk.set_proxy('http://127.0.0.1:9077/localproxy-16254641',user=None)
from nltk.corpus import stopwords
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))
from nltk.cluster.util import cosine_distance          #TO calculate cosine distance
import numpy as np                                     #TO create similarity matrix
import networkx as nx       
def read_article(text):
#     file = open(file_name, "r", encoding='utf-8')
#     filedata = file.readlines()                      #Reads all lines of .txt file and store the data in list
    article = text.split(". ")                #Seperates each line and removes '.'
    sentences = []
    
    #print (article)
    print ("\n\n")
    for sentence in article:
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    top_sentences = []

    # Step 1 - Read text anc split it
    sentences =  read_article(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
    #print (sentence_similarity_martix, "\n\n")

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    #print (sentence_similarity_graph, "\n\n")
    scores = nx.pagerank(sentence_similarity_graph)
    #print (scores, "\n\n")
    print ("Max. Number of points: ", len(scores),"\n\n")

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True) 
    #print (ranked_sentence)
    #print("Indexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
        print (" ".join(ranked_sentence[i][1]),".\n\n")
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    #print (summarize_text)
    return summarize_text

    # Step 5 - Offcourse, output the summarize texr
    #print("Summarize Text: \n", ". ".join(summarize_text))