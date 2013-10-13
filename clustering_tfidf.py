import gensim
import operator
import random
import copy
from gensim import corpora, models, similarities
from sets import Set
from collections import defaultdict
from topia.termextract import tag
from stemming.lovins import stem
import logging
import os,glob
import string
import re
import numpy
import nltk
#import porter

       
# Function to calculate the centroid of a cluster
def centroid(clust):
    global tfidf_docs
    n = len(tfidf_docs[0])
    cent = []
    
    for i in range(n) :
        temp = 0
        temp2 = []
        temp2.append(i)
        for doc in clust:
            temp += tfidf_docs[doc][i][1]
        
        temp = temp/len(clust)
        temp2.append(temp)
        cent.append(temp2)
    return (cent)                            

# Function to calculate the distance between 2 nodes/documents
def dist(n1,n2):
    temp = 0
    for i in range(len(n1)):
        temp += ((n1[i][1] - n2[i][1])*10)**2
    temp = temp**.5      
    temp = float("{0:.2f}".format(temp))  
    return (temp)

# Function to calculate the minimum separation of a cluster from remaining clusters
def min_sep(clust,all_clust):
    global tfidf_docs
    temp_all = copy.deepcopy(all_clust)
    
    all_centroids = []
    for item in temp_all:
        
        all_centroids.append(centroid(item))
        
    clust_centroid = centroid(clust) 
    distances = []  
    for centr in all_centroids:
        distances.append(dist(clust_centroid,centr)) 
    minsep = min(distances)
    return(minsep)
    
# Function to calculate Silhouette Coefficient of a cluster       
def s_coeff(clust,all_clust):
    global tfidf_docs
    temp_all = copy.deepcopy(all_clust)
    temp_all.remove(clust)
    all_centroids = []
    for item in temp_all:
        all_centroids.append(centroid(item))
        
    clust_centroid = centroid(clust) 
       
    
    coeff = 0                 
    temp = 0
    for doc in clust:
        distances = [] 
        
        for centr in all_centroids:
            distances.append(dist(tfidf_docs[doc],centr))
        b = min(distances)
        a = dist(tfidf_docs[doc],clust_centroid)   
        c = max(a,b)
        print " a,b -- %f,%f" %(a,b)
        if a == 0.0 and b == 0.0 :
            temp = 1
        else:    
            temp = (b - a)/c
        coeff += temp             
    coeff = coeff/(len(clust)) 
    return (coeff)    
    
# Function to Calculate average of the edges
def avg_edge(edges):
    n = len(edges)
    sum_edge = 0
    for item in edges:
        sum_edge += item[2]
        
    return (sum_edge/n)                

def optimize(final_clust) :

    print "Final Optimization"
    final_centroids = []
    for cluster in final_clust:
        temp = centroid(cluster)
        final_centroids.append(temp) 

    singles = []
    doubles = []
    for cluster in final_clust :
        if len(cluster) == 1 : singles.append(cluster)        
    
    while (len(singles) != 0):
        clust = singles[0]
        singles.remove(clust)
        distances = []
        for item in final_centroids:
            distances.append(dist(tfidf_docs[clust[0]],item))
        itself = distances.index(0.0)
        distances[itself] = max(distances) + 1
               
        min_d = min(distances)
        i = distances.index(min_d)
        
        c1 = clust
    
        c2 = final_clust[i]

        copy_final = copy.deepcopy(final_clust)
        sc1 = 1
        sc2 = s_coeff(c2,copy_final)     
        c3 = []
        for item in c2:
            c3.append(item)
        c3.append(c1[0])    
       
        copy_final2 = copy.deepcopy(final_clust)
       
        copy_final2.remove(c1)
    
        copy_final2.remove(c2)
        copy_final2.append(c3)   
        sc3 = s_coeff(c3,copy_final2)    
        uncombined_sc = (1 + sc2*len(c2))/(1 + len(c2))  
        print " (%r,%r) and %r -- %r and %r " %(c1,c2,c3,uncombined_sc,sc3) 
        
        if sc3 >= uncombined_sc or sc3 > .45:
            final_clust.remove(c1)
            final_clust.remove(c2)
            final_clust.append(c3) 
            if len(c2) == 1 : singles.remove(c2)
            final_centroids = []
            for cluster in final_clust:
                temp = centroid(cluster)
                final_centroids.append(temp)

    print "Final Clusters"        
    print final_clust


    print "Merging Doublets"                  
    for cluster in final_clust :
        if len(cluster) == 2 : doubles.append(cluster)        
    
    while (len(doubles) != 0):
        clust = doubles[0]
        doubles.remove(clust)
        clust_centr = centroid(clust)
        distances = []
        for item in final_centroids:
            distances.append(dist(clust_centr,item))
        itself = distances.index(0.0)
        distances[itself] = max(distances) + 1
               
        min_d = min(distances)
        i = distances.index(min_d)
    
        c1 = clust

        c2 = final_clust[i]

        copy_final = copy.deepcopy(final_clust)
        sc1 = s_coeff(c1,copy_final)
        sc2 = s_coeff(c2,copy_final)     
        c3 = []
        for item in c2:
            c3.append(item)
        for item in c1:        
            c3.append(item)    
           
        copy_final2 = copy.deepcopy(final_clust)
       
        copy_final2.remove(c1)
        
        copy_final2.remove(c2)
        copy_final2.append(c3)   
        sc3 = s_coeff(c3,copy_final2)    
        uncombined_sc = (sc1 + sc2*len(c2))/(2 + len(c2))  
        print " (%r,%r) and %r -- %r and %r " %(c1,c2,c3,uncombined_sc,sc3) 
    
        if sc3 >= uncombined_sc or sc3 > .45:
            final_clust.remove(c1)
            final_clust.remove(c2)
            final_clust.append(c3) 
            if len(c2) == 2 : doubles.remove(c2)
            final_centroids = []
            for cluster in final_clust:
                temp = centroid(cluster)
                final_centroids.append(temp)
    final_sc = []
    for item in final_clust :
        temp = copy.deepcopy(final_clust)
        final_sc.append(s_coeff(item,temp))   

    temp = []
    temp.append(final_clust)
    temp.append(final_sc)
    return (temp)

# Function to calculate the Cuts of a Graph
def clust_cut(in_clust,Edges) :

    n = len(in_clust)
    n2 = 1 # 5*n**3
    i = 0

    max_weight = 0
    max_cut = []
    for i in range(n2):  
        #print "iteration--%d" %i  
        #tempg = copy.deepcopy(docG)
        tempe = copy.deepcopy(Edges)
        tempn = n
        temp_cut = copy.deepcopy(in_clust)
    
        while tempn>2:
            e = random.randrange(0,len(tempe))
        
            #print e
            picked_edge = min(tempe, key  = lambda x : x[2])
            #picked_edge = tempe[e]
            #print picked_edge
            u = picked_edge[0]
            v = picked_edge[1]
            tempn -= 1
        
            #Merging set v to u
            j = 0
            for j in range(n):
                if temp_cut[j] == v:
                    temp_cut[j] = u
         
            # Removing edges between u and v
            for item in tempe:
                if (item[0] == u and item[1] == v) or (item[0] == v and item[1] == u) :
                    tempe.remove(item)

            # Modifying edges
            for item in tempe:
                if item[0]==v:
                    item[0] = u
                elif item[1]==v:
                    item[1] = u 
            # Removal of self loops
            for item in tempe:
                if item[0]==item[1]:
                    tempe.remove(item)                
            #print tempe
            #print temp_cut                     
                
        temp_weight = 0
        for item in tempe:
            temp_weight += item[2]
            
        if temp_weight > max_weight:
            max_weight = temp_weight
            u = tempe[0][0]                                                               
            v = tempe[0][1]
            group_u = []
            group_v = []
            k = 0
            for k in range(n):
                if temp_cut[k] == u :
                    group_u.append(clust[k])
                else:
                    group_v.append(clust[k])
            max_cut = []
            max_cut.append(group_u)
            max_cut.append(group_v)  
           # print max_cut
           # print max_weight

    return (max_cut,max_weight)    


print "Program Starts"
print "This program makes clusters of the given documents"


'''
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

a=[]  
temp=''
for line in documents :
      
    line=line.strip()
    line=line.lower()
    line=re.sub("<.*?>","",line)
    for c in string.punctuation:
        line=line.replace(c,'')
    a.append(line.split())    
    #a.extend(line.split())    
print "This is AAA"
print a
 
    
#--------------------------------------------------------------------------
#Extracting nouns
#--------------------------------------------------------------------------
documents2 = []

tagger = tag.Tagger()
tagger.initialize()
#documents = []
for item in a:
    temp=[]
    for i in range (0,len(item)) :

        x=item[i]          
        #text=nltk.pos_tag(nltk.Text(nltk.word_tokenize(x)))
        text=tagger(x)
             
        for noun in text:
              
            if(noun[1]=="NN" or noun[1]=="NNS" ):
                
                temp.append(noun[0])
                #temp+=noun[0]
                #temp+=' '
    documents2.append(temp)


print " After Noun Extraction"
print documents2  
'''

path='/home/abhiraj/COP/Test Corpora/Corpora 1'  
documents2 = [] 
tagger = tag.Tagger()
tagger.initialize()   
#--------------------------------------------------------------
#Reading From Corpus And Removing HTML Tags and punctuations
#-------------------------------------------------------------
a=[]
for file in glob.glob(os.path.join(path,'*.txt')):
    b = [] 
    temp=''
    for line in open(file) :
        print line      
        line=line.strip()
        line=line.lower()
        line=re.sub("<.*?>","",line)
        for c in string.punctuation:
            line=line.replace(c,'')
        line2 = line.split()
        for item in line2:
            b.append(item)
            
    a.append(b)            
        
        
for item in a:
    temp=[]
    for i in range (0,len(item)) :

        x=item[i]          
        #text=nltk.pos_tag(nltk.Text(nltk.word_tokenize(x)))
        text=tagger(x)
             
        for noun in text:
              
            if(noun[1]=="NN" or noun[1]=="NNS" ):
                
                temp.append(noun[0])
                #temp+=noun[0]
                #temp+=' '
    documents2.append(temp)
print "Here are documents after noun extraction"
for item in documents2:
    print item  
        
stoplist=[]
    
f=open('stopword.txt')

for line in f :

    line=line[0:-1]
    stoplist.append(line)
f.close()
    
#stoplist = set('for a of the and to in'.split())

texts = [[word for word in document if word not in stoplist]
          for document in documents2]


#texts = [[word for word in document.lower().split() if word not in stoplist]
 #         for document in documents]    
#print "Stopwords removed"
#print texts          


#--------------------------------------------------------------------------
#Stemming
#--------------------------------------------------------------------------
'''
for i in texts:
    
    for j in range (0,len(i)): 
        print len(i)
        print j
        print i[j]       
        k=stem(i[j])
        i[j]=k
        
print "After Stemming"

print texts   
'''     
        
all_tokens = sum(texts, [])
tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)
#print "once words"
#print tokens_once
texts = [[word for word in text if word not in tokens_once]
         for text in texts]
 
   
# Making the dictionary of relevant keywords
dictionary = corpora.Dictionary(texts)
             
item = 0
n = len(dictionary)
print '\nThis is the Dictionary of words in our Corpora:\n' 
'''
while item < n :
    print "%r:%r" %(item,dictionary[item])
    item += 1
'''
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

tfidf_docs_temp = []
for doc in corpus_tfidf :
    tfidf_docs_temp.append(doc)
    
num_t = len(dictionary)   
tfidf_docs = []

print " Creating tfidf Matrix"
for i in range(len(tfidf_docs_temp)):
    temp =[]
    k = 0
    for j in range(num_t):
    
        temp_var = []
        temp_var.append(j)
        if k < len(tfidf_docs_temp[i]) :
            if j == tfidf_docs_temp[i][k][0] :
                          
                temp_var.append(tfidf_docs_temp[i][k][1])
                temp.append(temp_var) 
                k += 1 
            else :
            
                temp_var.append(0)
                temp.append(temp_var)   
        else :
            temp_var.append(0)
            temp.append(temp_var)                         
    tfidf_docs.append(temp)     

i = 0
n = len(tfidf_docs)

while i < n : 
    print "%d==>" %(i+1),
    for item in tfidf_docs[i] :
        print "(%r,%.5f)" %(item[0],item[1]),
    print ""    
    i = i+1
    
i=0
docG = []
print "VALUE of n==%d" %n
for i in range(n):
    adj = []
    j = 0
    
    for j in range(n):
        if j==i : continue
        else : 
            sq_sum = 0
            k = 0
            for k in range(num_t) :
                sq_sum += ((tfidf_docs[j][k][1] - tfidf_docs[i][k][1])*10)**2
            edge = []
            edge.append(i)
            edge.append(j)
            root_sq_sum = sq_sum**.5
            root_sq_sum = float("{0:.2f}".format(root_sq_sum))
            edge.append(root_sq_sum)            
        
        adj.append(edge)
    docG.append(adj)                

for item in docG:    
    print item
    
Edges = []
i=0
j=0


for i in range(n):
    for item in docG[i]:
        if i < item[1]:
            Edges.append(item)   
            
# Removing edges of zero weight
for item in Edges:
    if item[2] == 0.0: item[2] = 1.01             
print "Here are all the edges:"        
print Edges 
num_e = len(Edges)           

init_clust = []
final_clust =[]
i = 0
for i in range(n):
    init_clust.append(i)
    

cut_ret = []
temp_clust =[]
s_coeffs = []
s_coeffs.append(0.0)
temp_clust.append(init_clust)
#temp_clust = copy.deepcopy(init_clust)

# Main Part of the Algorithm
# Keep on dividing graph into better cuts and evaluate them to calulate performance
# Stop when appropriate clusters are formed
while(True) :
    print "Temp Clusters"
    print temp_clust
    print s_coeffs
    if len(temp_clust) == 0:
        break
    else:       
        clust = temp_clust[0]
        temp_clust.remove(clust)
        t_edges = []
        for item in Edges:
            if (item[0] in clust) and (item[1] in clust) :
                t_edges.append(item)
                
        #t_edges = reduce_edges(t_edges)                
        print "Dividing",
        print clust
        
        # Special condition cheking for clusters of 2 documents
        if len(clust) == 2 :
            s_coeffs.remove(s_coeffs[0])
            minsep_temp = copy.deepcopy(temp_clust)
            if len(final_clust) != 0:
                for item in final_clust:
                    minsep_temp.append(item)
            temp1 = []
            temp2 = []
            temp1.append(clust[0])
            temp2.append(clust[1])                    
            d = dist(tfidf_docs[clust[0]],tfidf_docs[clust[1]])
            b1 = min_sep(temp1,minsep_temp)
            b2 = min_sep(temp2,minsep_temp)  
            b = (b1+b2)/2   
            print "b1,b2 ==> %f,%f " %(b1,b2)       
            print "min separation before ==> %f, dist b/w docs ==> %f)" %(b,d)
            if b > d:
                final_clust.append(clust)
            else:
                 
                final_clust.append(temp1)
                final_clust.append(temp2)
        
        # for normal clusters with more than 2 documents
        else : 
            # dividing the cluster into cuts
            cut_ret = clust_cut(clust,t_edges)
            #print "Divided"
            temp1 = cut_ret[0][0]
            temp2 = cut_ret[0][1] 
        
            # Calculating silhouette coefficients of new cuts and comparing with parent cluster
            sil_coeff_temp = copy.deepcopy(temp_clust)
            if len(final_clust) != 0:
                for item in final_clust:
                    sil_coeff_temp.append(item)        
            sil_coeff_temp.append(temp1)
            sil_coeff_temp.append(temp2)        
            print "Evaluating",
            print temp1
            print "Sil Coeff"
            if len(temp1) == 1:
                sc1 = 1
            else:                
                sc1 = s_coeff(temp1,sil_coeff_temp)
            print sc1
        
                    
            print "Evaluating",
            print temp2
            print "Sil Coeff"
            if len(temp2) == 1:
                sc2 = 1
            else:
                sc2 = s_coeff(temp2,sil_coeff_temp)
            print sc2
            
            parent_sc = s_coeffs[0]
            s_coeffs.remove(s_coeffs[0])
            if (len(temp1) > 1 and len(temp2) > 1):
                combined_sc = (sc1*len(temp1) + sc2*len(temp2))/(len(temp1) + len(temp2))
                print combined_sc
        
                
        
                if combined_sc < parent_sc and (sc1 > .25 or sc2 > .25) :
                    if (sc1 > 0.25 and sc2 > 0.25) :
                        final_clust.append(clust)
                    elif  (sc1 < 0.25 and sc2 > 0.25)   :
                        final_clust.append(temp2)
                        temp_clust.append(temp1)
                        s_coeffs.append(sc1)
                    else : 
                        final_clust.append(temp1)
                        temp_clust.append(temp2)
                        s_coeffs.append(sc2)
        
                else   :
                    if len(temp1) == 1:
                        final_clust.append(temp1)
                    else:    
                        temp_clust.append(temp1)
                        s_coeffs.append(sc1)
                    if len(temp2) == 1:
                        final_clust.append(temp2)
                    else:     
                        temp_clust.append(temp2)
                        s_coeffs.append(sc2)
                        
            elif (len(temp1) == 1 and len(temp2) > 1):
                minsep_temp = copy.deepcopy(temp_clust)
                if len(final_clust) != 0:
                    for item in final_clust:
                        minsep_temp.append(item)
                b = min_sep(clust,minsep_temp)
                d = dist(tfidf_docs[temp1[0]],centroid(temp2))                
                print "min separation before ==> %f, dist b/w cuts ==> %f)" %(b,d)
                
                if 0.75*b > d:
                    final_clust.append(clust)
                else:
                    final_clust.append(temp1)
                    temp_clust.append(temp2)
                    s_coeffs.append(sc2)
            elif (len(temp2) == 1 and len(temp1) > 1):
                if len(clust) == len(tfidf_docs):
                    b = 0
                else:
                    minsep_temp = copy.deepcopy(temp_clust)
                    if len(final_clust) != 0:
                        for item in final_clust:
                            minsep_temp.append(item)
                    b = min_sep(clust,minsep_temp)
                d = dist(tfidf_docs[temp2[0]],centroid(temp1))                
                print "min separation before ==> %f, dist b/w cuts ==> %f)" %(b,d)
                
                if 0.75*b > d:
                    final_clust.append(clust)
                else:
                    final_clust.append(temp2)
                    temp_clust.append(temp1)
                    s_coeffs.append(sc1)                    
                
        print "Final Clust",
        print final_clust            

final_opt = optimize(final_clust)
final_clust = final_opt[0]
final_sc = final_opt[1]
print final_clust
print final_sc
  
'''
print "\n\tHere are the TF-IDF values of the documents\n"    
i = 0
n = len(tfidf_docs)

while i < n : 
    print "%d==>" %i,
    for item in tfidf_docs[i] :
        print "(%r,%.5f)" %(item[0],item[1]),
    print ""
    i += 1

tfidf_table = {}

    # Initialisation of the table    
for item in dictionary : 
    tfidf_table[dictionary[item]] = []    
    i = 0
    while i < doclen : 
        tfidf_table[dictionary[item]].append(0) 
        i += 1  

i = 0
while i < doclen :

    for item in tfidf_docs[i] :  
        key = item[0]
        word = dictionary[key]
        tfidf_table[word][i] = item[1] 
    i += 1               

print "\n\tHere is the TF-IDF Table--\n"
for word in tfidf_table:
    #print "%r:%r" %(item,tfidf_table[item])
    print "%s->" %word ,   
    for item in tfidf_table[word]:
        print "%.5f " %item ,
    print ' '

cnum = raw_input("\nEnter the number of clusters :")    
clusters = []
centroid = []
i = 0
n = cnum
while i<2 :
    print "Enter documents for this cluster"
    x = raw_input()
    x = x.split()
    clusters.append(x)
    i += 1
    
print clusters  


for clust in clusters :
    lenc = len(clust)
    clust_mean = []
    j = 0
    while j < len(tfidf_table) :
        sumword = 0
        word = dictionary[j]
        for docu in clust :
            sumword += tfidf_table[word][int(docu)]
        sumword = sumword/lenc            
        clust_mean.append(sumword)
        j += 1
    centroid.append(clust_mean)

print "\n\tHere are the centroids for each cluster\n"    
for item in centroid:
    for value in item :
        print "%.5f" %value,
    print ""

    #main Centroid for corpora
i = 0
main_centroid = []

while i < len(tfidf_table):
    temp = 0
    for item in centroid:
        temp += item[i]
    main_centroid.append(temp/len(centroid))
    i += 1

print "\nHere is the Main Centroid for the Corpora"   
for item in main_centroid:
    print "%.5f" %item,
    
print ""    
                
dist = []
i = 0
for clust in clusters:
    clust_dist = []
    for docu in clust:
        j=0
        e_dist = 0
        while j < len(tfidf_table) :
            word = dictionary[j]
            temp = (tfidf_table[word][int(docu)] - centroid[i][j])**2     
            e_dist += temp   
            j += 1   
        e_dist = e_dist**.5  
        clust_dist.append(e_dist)
        
    dist.append(clust_dist) 
    i += 1         
        
print "\nHere are the distances of documents from centroids within clusters\n"  
for item in dist:
    for value in item:
        print "%.5f" %value,
    print""      
    
cohesion = []
for item in dist:
    temp = 0
    for value in item:
        temp += value
    temp = temp/len(item)
    cohesion.append(temp)
    

print "\nHere are the Cohesions for each cluster\n"
i = 1   
for item in cohesion:
    print "Cohesion for cluster %d -- %.5f" %(i,item)
    i += 1
    
sep_dist = []
for clust in centroid:
    temp_dist = 0
    i = 0
    while i < len(clust):
        temp_dist += (clust[i] - main_centroid[i])**2
        i += 1
    temp_dist = temp_dist**.5
    sep_dist.append(temp_dist)
        
print "\nHere are the Separations for each cluster\n"
i = 1   
for item in sep_dist:
    print "Separation for cluster %d -- %.5f" %(i,item)
    i += 1       
        
 '''       



      
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
