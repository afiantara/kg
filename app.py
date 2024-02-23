## for data
import pandas as pd  #1.1.5
import numpy as np  #1.21.0

## for plotting
import matplotlib.pyplot as plt  #3.3.2

## for text
#import wikipediaapi  #0.5.8
import nltk  #3.8.1
import re   

## for nlp
import spacy  #3.5.0
from spacy import displacy
import textacy  #0.12.0

## for graph
import networkx as nx  #3.0 (also pygraphviz==1.10)

## for timeline
import dateparser #1.1.7

def network_graph(dtf,filter=''):
    ## create full graph
    tmp=dtf
    if filter !='':
        tmp = dtf[(dtf["entity"]==filter) | (dtf["object"]==filter)]
    
    print(tmp)

    G = nx.from_pandas_edgelist(tmp, source="entity", target="object", 
                                edge_attr="relation", 
                                create_using=nx.DiGraph())


    ## plot
    plt.figure(figsize=(15,10))

    pos = nx.spring_layout(G, k=1)
    node_color = "skyblue"
    edge_color = "black"

    nx.draw(G, pos=pos, with_labels=True, node_color=node_color, 
            edge_color=edge_color, cmap=plt.cm.Dark2, 
            node_size=2000, connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_edge_labels(G, pos=pos, label_pos=0.5, 
                            edge_labels=nx.get_edge_attributes(G,'relation'),
                            font_size=12, font_color='black', alpha=0.6)
    plt.show()

def extract_entities(doc):
    a, b, prev_dep, prev_txt, prefix, modifier = "", "", "", "", "", ""
    for token in doc:
        if token.dep_ != "punct":
            ## prexif --> prev_compound + compound
            if token.dep_ == "compound":
                prefix = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
            
            ## modifier --> prev_compound + %mod
            if token.dep_.endswith("mod") == True:
                modifier = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
            
            ## subject --> modifier + prefix + %subj
            if token.dep_.find("subj") == True:
                a = modifier +" "+ prefix + " "+ token.text
                prefix, modifier, prev_dep, prev_txt = "", "", "", ""
            
            ## if object --> modifier + prefix + %obj
            if token.dep_.find("obj") == True:
                b = modifier +" "+ prefix +" "+ token.text
            
            prev_dep, prev_txt = token.dep_, token.text
    
    # clean
    a = " ".join([i for i in a.split()])
    b = " ".join([i for i in b.split()])
    return (a.strip(), b.strip())


# The relation extraction requires the rule-based matching tool, 
# an improved version of regular expressions on raw text.
def extract_relation(doc, nlp):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    p1 = [{'DEP':'ROOT'}, 
          {'DEP':'prep', 'OP':"?"},
          {'DEP':'agent', 'OP':"?"},
          {'POS':'ADJ', 'OP':"?"}] 
    matcher.add(key="matching_1", patterns=[p1]) 
    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]] 
    return span.text

def init():
    with open('data/asuransi.txt', encoding='cp1252') as f:
        textz = f.read().replace('\n', ' ')  # read the file and remove new lines
        textz = ' '.join(textz.split())  # split the text into words, remove unnecessary spaces, and join the words with a single space

    # Load the spaCy model
    nlp = spacy.blank('id')
    #nlp = spacy.load("en_core_web_sm")
    # Read the text file
    text = textz
    print(text)
    # Extract the keywords using spaCy
    from spacy.lang.id.stop_words import STOP_WORDS
    doc = nlp(text)
    print(doc)

    str_template = '{:>15} {:>10} {:>10} {:>10} {:>10} {:>10}'
    print(str_template.format('token', 'is_lower', 'is_title', 'is_upper', 'is_digit', 'is_punct'))
    for token in doc:
        print(str_template.format(str(token),
            str(token.is_lower),
            str(token.is_title),
            str(token.is_upper),
            str(token.is_digit),
            str(token.is_punct)))

    from spacy.lang.id import LOOKUP
    import random
    lemma_as_list = list(LOOKUP.items())
    samples = random.choices(lemma_as_list, k=20)
    for k, v in samples:
        print(f'{k}: {v}')

    # from text to a list of sentences
    #lst_docs = [sent for sent in doc.sents]
    #print("tot sentences:", len(lst_docs))
    
    # take a sentence
    #i = 3
    #lst_docs[i]
    
    #for token in lst_docs[i]:
    #    print(token.text, "-->", "pos: "+token.pos_, "|", "dep: "+token.dep_, "")

    '''
    dic = {"id":[], "text":[], "entity":[], "relation":[], "object":[]}

    for n,sentence in enumerate(lst_docs):
        lst_generators = list(textacy.extract.subject_verb_object_triples(sentence))  
        for sent in lst_generators:
            subj = "_".join(map(str, sent.subject))
            obj  = "_".join(map(str, sent.object))
            relation = "_".join(map(str, sent.verb))
            
            dic["id"].append(n)
            dic["text"].append(sentence.text)
            dic["entity"].append(subj)
            dic["object"].append(obj)
            dic["relation"].append(relation)

    ## create dataframe
    dtf = pd.DataFrame(dic)
    print(dtf["entity"].value_counts().head())
    filter='Silicon Valley'
    network_graph(dtf,filter)
    '''
if __name__=="__main__":
    init()