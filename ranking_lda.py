import gensim
import operator
from gensim import corpora, models, similarities

print "Program Starts"
print "This program Ranks the documents within clusters from a given document set"

filename = raw_input("Enter the name of input file:") 
print filename
fp = open(filename,'r')
documents = []

while True : 
    x = fp.readline().strip('\n')
    if len(x) == 0 :
        break
    else : 
        documents.append(x)        

fp.close()
print '\nSo here is our Document set\n'

for item in documents :
    print item
    
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist]
          for document in documents]    
          
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
texts = [[word for word in text if word not in tokens_once]
         for text in texts]
         
dictionary = corpora.Dictionary(texts)
             
item = 0
n = len(dictionary)
print '\nThis is the Dictionary of words in our Corpora:\n' 
while item < n :
    print "%r:%r" %(item,dictionary[item])
    item += 1

corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]





'''
thresh = 0.6
clust = []
i = 0
while i < num :
    ci = []
    j = 0
    while j < n :
        if lda_docs[j][i][1] >= thresh :
            temp = []
            temp.append(j+1)
            temp.append(lda_docs[j][i][1])
            ci.append(temp)
        j += 1            
    clust.append(ci)
    i += 1        
'''
print "\nEnter the number of clusters :" ,
cnum = input() 

lda = gensim.models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=cnum)
corpus_lda = lda[corpus_tfidf]

lda_docs = []
for doc in corpus_lda :
    lda_docs.append(doc)
i = 0
n = len(lda_docs)

while i < n : 
    print "%d==>" %(i+1),
    for item in lda_docs[i] :
        print "(%r,%.5f)" %(item[0],item[1]),
    print ""    
    i = i+1
 
clust = []
i = 0

while i < cnum :
    print "Enter documents cluster %d" %(i+1)
    x = raw_input()
    x = x.split()
    for item in x :
        item = int(item)
    clust.append(x)
    i += 1
print clust
i = 0    
while i < len(clust) : 
    j = 0
    while j < len(clust[i]):
        temp = []
        temp.append(clust[i][j])
        temp.append(lda_docs[int(clust[i][j])][i][1])
        clust[i][j] = temp
        j += 1
    i += 1        

print "\nThe Clusters are--"
i = 1
while i <= cnum :
    print "\nCluster %r -- " %i
    for item in clust[i-1]:
        print "%s--%.5f" %(item[0],item[1])
    i += 1
    print ""
print ""
#print clust
ranked_clust = []
i = 0
while i < cnum :
    temp = sorted(clust[i], key = lambda x : -x[1])
    ranked_clust.append(temp)
    i += 1
    
print "\nAfter Ranking the clusters are--"
i = 1
while i <= cnum :
    print "\nRanked Cluster %r -- " %i,
    for item in ranked_clust[i-1]:
        print item[0],
    i += 1











