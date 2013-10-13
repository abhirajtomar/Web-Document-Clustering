print "TF-IDF Table:"

fp = open("datainp.txt",'r')
documents = []

tfidf_table = {}

while True :
    x = fp.readline().strip('\n') 
    if len(x) == 0 :
        break
    else :
        tfidf_table[x] = []
        i = 1
        #temp = []
        while i <= 10:
            y = fp.readline().strip('\n')
            tfidf_table[x].append(float(y))
            i += 1
           
for item in tfidf_table:
    print "%s->" %item ,
    for docs in tfidf_table[item] :
        print "%.5f " %docs ,                      
    print ' '                
   
dictionary = []
for item in tfidf_table:
    dictionary.append(item)
    
print "Dictionary of words--- %r" %dictionary   
   
print "\nEnter the number of clusters :"    
cnum = input()    
clusters = []
centroid = []
i = 0
while i<cnum :
    print "Enter documents for this cluster"
    x = raw_input()
    x = x.split()
    clusters.append(x)
    i += 1
    
print clusters  

print "\n\tNow calculating terms for cohesion within each cluster"
for clust in clusters :
    lenc = len(clust)
    clust_mean = []
    for word in tfidf_table :
        sumword = 0
        for docu in clust :
            sumword += tfidf_table[word][int(docu)]
        sumword = sumword/lenc            
        clust_mean.append(sumword)
    centroid.append(clust_mean)

print "\n\tHere are the centroids for each cluster\n"    
c = 0
for item in centroid:
    c += 1
    print "Cluster--%d" %c
    j = 0
    for value in item :
        print "%s-->" %dictionary[j],
        print "%.5f" %value
        j += 1
    print ""
                
dist = []
            
i = 0
while i < 2:
    clust_dist = []
    j = 0
    while j < len(tfidf_table) :
        
        k = 0
        dsum = 0
        while k < len(clusters[i]) :
            d = centroid[i][j] - tfidf_table[dictionary[j]][k]
            if d < 0 : d = -d 
            
            dsum += d
            k += 1
        clust_dist.append(dsum)
        j += 1
    dist.append(clust_dist)
    i += 1

print "\n\tHere are the distances from the centroid values within each cluster\n"
c = 0     
for item in dist:
    c += 1
    print "Cluster--%d" %c
    j = 0
    for value in item :
        print "%s-->" %dictionary[j],
        print "%.5f" %value
        j += 1
    print ""            
                                
print "\n\tNow calculating the separation between clusters"                             

sep_dist = []
i = 0
while i < len(centroid[0]):
    sep_dist.append(0)
    i += 1
i = 1
while i < len(centroid) :
    j = 0
    while j < len(centroid[0]) :
        d = centroid[0][j] - centroid[i][j]
        if d < 0 : d = -d 
        sep_dist[j] += d
        j += 1        
    i += 1
print "\n\tHere are the separations\n"
j = 0
for value in sep_dist :
    print "%s-->" %dictionary[j],
    print "%.5f" %value
    j += 1
    
