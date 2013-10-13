import gensim
from gensim import corpora, models, similarities


        # Document set containing 10 documents, topics - fruits, football, algorithms
documents =  ["i like to eat spinach and apple",
              "i had oranges and toast for breakfast",
              "Football is the most popular sport in the world",
              "BFS and DFS are graph algorithms",
              "Oranges have a number of varieties",   
              "Football players have lots of fruits in their diet",
              "Strategies in football are based on algorithms these days",
              "Higher algorithms involve advanced data structures",
              "Football is also known as soccer",
              "Apple has caffiene in it"]
              
              # remove common words and tokenize
              
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]
             

               # remove words that appear only once
               
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]

            # Creating a dictionary from the processed set of words
dictionary = corpora.Dictionary(texts)
print dictionary

            # outputs the final set of words in the dictionary
for item in dictionary:
    print dictionary[item]


            # the documents are processed based on the words in the dictionary
corpus = [dictionary.doc2bow(text) for text in texts]

print "\n"

            # TF-IDF transformation
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

            # LDA transformation
lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=15)
corpus_lda = lda[corpus_tfidf]

            # Outputs the various documents in terms of their relation to the lda topics
print "\n The documents transformed into LDA model:"
for doc in corpus_lda:
    print "\n %r" %doc
    
            # Submitting Query 
query = "i have my oranges and football"
print "\n\nThis is the query %r: " %query

            # transforming query into bag of words form
vec_bow = dictionary.doc2bow(query.lower().split())
print "\nQuery in bag of words form:"
print vec_bow  
            
            # transformation of query in LDA model
vec_lda = lda[vec_bow]
print "\n Query transformed into LDA model"
print vec_lda

            # indexing the query for matching with the document set
index = similarities.MatrixSimilarity(lda[corpus])

sims = index[vec_lda]

            # printing the relevance of documents with the query
print "\n Relevance of documents with query"
print sims
print list(enumerate(sims))

            # now sorting the relevances to give final ranking order
print "\n Final ranking of documents"
sims = sorted(enumerate(sims),key = lambda item: -item[1])
print sims









        
