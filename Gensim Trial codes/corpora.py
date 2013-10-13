import gensim
from gensim import corpora, models, similarities
print "Program Starts"
documents2 = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
             
documents = ["large singular value computations",
                "software library for the space singular value decomposition",
                "introduction to modern information retrieval",
                "using linear algebra for intelligent information retrieval",
                "matrix computations",
                "Singular value analysis of cryptograms",
                "automatic information organization" ]             
             
# remove common words and tokenize
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]

# remove words that appear only once
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]

print texts

dictionary = corpora.Dictionary(texts)
#dictionary.save('/deerwester.dict') # store the dictionary, for future reference
print dictionary

print dictionary.token2id
'''
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print new_vec'''

corpus = [dictionary.doc2bow(text) for text in texts]

print "\n"
print corpus

tfidf = models.TfidfModel(corpus) # Step1-- Initialization of a model

doc_bow = [(0,1),(1,1)]
print tfidf[doc_bow] # Step2--- Using models to transform vectors

corpus_tfidf = tfidf[corpus]
for doc in corpus_tfidf:
    print doc
    
print "Starting LDA transformation"
    
lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=5)
'''lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)
corpus_lda = lda[corpus_tfidf]'''
print "LDA transformation Done"

#models.LsiModel.print_topics(lsi)
print "These are the LDA topics-"
'''lda.print_topics()
print ""
for doc in corpus_lda:
    print doc'''
#lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=2)
corpus_lsi = lsi[corpus_tfidf]
print ""
for doc in corpus_lsi:
    print "yo"
    print doc

'''
doc = "Graph minors A computer"
vec_bow = dictionary.doc2bow(doc.lower().split())

print vec_bow

vec_lda = lda[vec_bow]

print vec_lda
'''

'''
index = similarities.MatrixSimilarity(lda[corpus])

#sims = index[vec_lda]
sims = index
print list(enumerate(sims))
print "Now in sorted order"
sims = sorted(enumerate(sims))    #,key = lambda item: -item[1]
print sims
'''









