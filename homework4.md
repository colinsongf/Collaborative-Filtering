##Homework4
<br>
AndrewId: xiaoxul 

Name: Xiaoxu Lu

---
###0.Statement of Assurance 
I certify that everything I submit is writen by myself and only by myself.
###1.Corpus Exploration 
####1.1 Basic statistics
 
Statistics | |
-----------|---|
the total number of movies|5392
the total number of users|10916
the number of times any movie was rated '1'|53852
the number of times any movie was rated '3'|260055
the number of times any movie was rated '5'|139429
the average movie rating across all users and movies| 3.381

For user ID 4321 | |
-----------|------|
the number of movies rated|73
the number of times the user gave a '1' rating|4
the number of times the user gave a '3' rating|28
the number of times the user gave a '5' rating|8
the average movie rating for this user|3.151


For movie ID 3| |
-----------|----|
the number of users rating this movie|84
the number of times the user gave a '1' rating|10
the number of times the user gave a '3' rating|29
the number of times the user gave a '5' rating|1
the average rating for this movie|2.524


####1.2 Nearest Neighbors 

||Nearest Neighbors|
|--|-----------------|
|Top 5 NNs of user 4321 in terms of dot product similarity|980 2586  551 3760   90|
|Top 5 NNs of user 4321 in terms of cosine similarity|8202 7700 3635 9873 8497|
|Top 5 NNs of movie 3 in terms of dot product similarity|1466 3688 3835 2292 4927|
|Top 5 NNs of movie 3 in terms of cosine similarity|5370 4857 5065 5391 4324|

###2.Basic Rating Algorithms 
####2.1 User-user Similarity
Rating Method|Similarity Metric|K|RMSE|Runtime(sec)*
-------------|-----------------|-|----|------------
Mean|Dot product|10|1.00259912228|53.3912s
Mean|Dot product|100|1.00666074226|72.3590s
Mean|Dot product|500|1.04298657984|138.3591s
Mean|Cosine|10|1.06306036204|118.3650s
Mean|Cosine|100|1.06200633237|127.7857s
Mean|Cosine|500|1.07535601069|182.9375S
Weighted|Cosine|10|1.06299574024|120.8656s
Weighted|Cosine|100|1.06191618427|159.6333s
Weighted|Cosine|500|1.07513046072|305.9817s

####2.2 Movie-movie Similarity
Rating Method|Similarity Metric|K|RMSE|Runtime(sec)*
-------------|-----------------|-|----|------------
Mean|Dot product|10|1.02072327298|17.8319s
Mean|Dot product|100|1.04679255029|131.8365s
Mean|Dot product|500|1.11089982645|598.7017s
Mean|Cosine|10|1.02476859176|18.9528s
Mean|Cosine|100|1.04977274366|147.6397s
Mean|Cosine|500|1.12376452635|893.2365s
Weighted|Cosine|10|1.02258570517|40.9286s
Weighted|Cosine|100|1.0390252412|349.1672s
Weighted|Cosine|500|1.1226454222|1143.4752s


####2.3.1 Movie-rating normalization
Rating Method|Similarity Metric|K|RMSE|Runtime(sec)*
-------------|-----------------|-|----|------------
Mean|Dot product|10|1.00659815728|57.5093s
Mean|Dot product|100|0.999206481309|79.3911s
Mean|Dot product|500|1.034736352537|154.3934s
Mean|Cosine|10|1.0054750914|116.6785s
Mean|Cosine|100|1.00513275711|135.7514s
Mean|Cosine|500|1.00503846352|203.2477s
Weighted|Cosine|10|1.00531984207|118.8196s
Weighted|Cosine|100|1.00500336293|167.7712s
Weighted|Cosine|500|1.00384724736|296.7510s


####2.3.2 Detailed Descriptions for normalize algorithm
I tried both normalization: pcc-like bias elimination on movies for memory-based CF(user-user similarity) and pcc-like bias elimination on users for model-based CF(movie-movie simialrity). My experiment shows that eliminating bias on movies are more helpful than doing the same thing on users. The latent reason for this situation could be that, for the dataset given, users' preference are not remarkably compared to movies' quality. There are not a large group of users being so generous(always give high ratings) while others being so strict(always give low ratings). On the other hand, movies differ a lot more than users. There are movies that are highly rated and there are movies that are not liked by users. The bias of movies is greater than that of users. So I choose to normalize on movie.

The algorithm for normalized rating:

for movie j, user i (from 0 to n), get the movie's mean and standard deviation, the normalized rating for (i,j) is:

![f1 icon](f1.png =150x)

Then when computing the predictions, using the mean and standard deviation to to predict for (i,j):

![f2 icon](f2.png =162x)
 


####2.4.1 Bipartite Clustering Information
Running time of bipartite clustering in seconds: 2887.0947s

Total number of user clusters: 	3000

Total number of item clusters: 1500
			
How did you pick the number of clusters?

As the CF algorithm gets a max k as 500, it's better that Bipartite clustering performs on both clusters greater than 500. However, I still tried different combinations of k.
(for item similarity, k1 for user clusters and k2 for movie clusters)

(k1, k2)|RMSE|Runtime
---------|----|-------
(300, 100)|1.1609|196.4826s
(500, 250)|1.1557|303.0831s
(1000, 500)|1.1320|635.5301s
(3000, 1500)|1.1004|2887.0947s

It's obvious that with the value of k1 and k2 increases, the RMSE decreases correspondingly. A larger value benefits the performance but if we make k equal to the number of instances, then the clustering is meaningless as each instance of itself is a cluster. The reason for performing bipartite clustering or say clustering, is trying to borrow or share information from similar instances.


####2.5 User-user Similarity
Rating Method|Similarity Metric|K|RMSE|Runtime(sec)*
-------------|-----------------|-|----|------------
Mean|Dot product|10|1.05526291825|17.6088s
Mean|Dot product|100|1.0872645242|32.7653s
Mean|Dot product|500|1.1028375439|100.2376s
Mean|Cosine|10|1.06455804997|18.5639s
Mean|Cosine|100|1.0936573927|54.9283s
Mean|Cosine|500|1.1000928365|127.9386s
Weighted|Cosine|10|1.06442074247|26.9668s
Weighted|Cosine|100|1.0827162221|63.8745s
Weighted|Cosine|500|1.1284764643|142.7145s
####2.6 Movie-movie Similairty
Rating Method|Similarity Metric|K|RMSE|Runtime(sec)*
-------------|-----------------|-|----|------------
Mean|Dot product|10|1.10043844647|21.7322s
Mean|Dot product|100|1.04679255029|127.7810s
Mean|Dot product|500|1.11089982645|344.9163s
Mean|Cosine|10|1.10379203008|21.2792s
Mean|Cosine|100|1.1129385639|176.8721s
Mean|Cosine|500|1.1247301929|346.8264s
Weighted|Cosine|10|1.10074951608|45.7839s
Weighted|Cosine|100|1.1294864673|298.8620s
Weighted|Cosine|500|1.1305837622|682.7452s
###3.Analysis of results 
##### Observation 0: Value of k

For whatever similarity metric selected and whatever weighting scheme used, even with pcc like bias elimination, or bipartite clustering, K = 10 always give the best RMSE. When k grows, the RMSE grows which shows the performance decreases. It's not hard to understand, for example, given a query user, it's easier to find 10 similar users than to find 500 similar users. Trying to get a large set of similar users is unwise as there might be a large subset of it are not really similar to the query user. They are selected as k is so large. This is also true when it comes to movies. So a reasonable k for k nearest neighborhood on user-user similarity and movie-movie similarity is important. With some experiments, I find that when k falls into [25,30], the algorithm could achieve a relatively low RMSE.

##### Observation 1: Feature: user similarity vs. movie similarity
User simialrity is a little better than the movie similarity. This is not hard to find given the same similairty metric and the value of k. As I have discussed in the pcc-like method, there are not a large group of users being so generous(always give high ratings) while others being so strict(always give low ratings). On the other hand, movies differ a lot more than users. There are movies that are highly rated and there are movies that are not liked by users. As a result, for the dataset, user similarity is a better indicator than movie similarity.

##### Observation 2: Similaity metric: dot product vs. cosine similarity
Given the same value of k, the same similarity metric, for user-user similarity and movie-movie similarity, either choosing the dot product or choosing cosine similarity does not differ a lot. Cosine similarity only differs from dot product that it goes through a normalization process. For the dataset given, it's so sparse that such normalization does little work to help us revise the prediction. So these 2 metric are both good indicators for measuring similarity. 

##### Observation 3: Weighting scheme: mean vs. weighted sum

Given the same value of k, on cosine simialrity, for user-user similarity and movie-movie similarity, either choosing mean or choosing weight sum as weightng scheme does not differ a lot. These could be caused by yhe same reason discussed in Observation 2, the matrix being so sparse.

##### Observation 4: Bias elimination

See 2.3.2 for analysis.
Given the same parameter and metric setting, elimination on bias sometimes help to achieve lower RMSE and better performance while sometimes not. When combined with a reasonable value of k and good metric for similarity computation and weighting scheme, pcc like bias elimination could be better and better. There could be a best combination of all the parameter setting based on this method given the dataset. Actually, based on my implementation, the best RMSE owing too the elimination of movie bias.

##### Observation 5: Bipartite clustering
Given the same parameter and metric setting, bipartite clustering does not help on better prediction. This is caused by choosing a large k for bipartite clustering. This actually gives similar feedback as choosing a large k for CF. Borrowing or sharing information in clusters where the items in it are not that similar is not helpful at all. A better decision for selecting k for bipartite clustering could benefit the performance -- this is my intuition.


###4.The software implementation 
####4.1 A description of what you did to preprocess the dataset to make your implementations easier or more efficient.

####4.2 A description of major data structures (if any); any programming tools or libraries that you used.
For 4.1 and 4.2:

When I talk about memory-based CF, I refer to user-user similarity memory-based CF. When I talk about model-based CF, I refer to movie-movie similarity model-based CF. The truth is that model-based one could also be used for user and memory based one could also be used for movie. Just make it clear here.

When reading from the queries(dev.csv), for memory-based and model based CF, there are two different stories. For memory-based one, each time there is a query coming, the similaity bewtween the user in this query and all users would be computed for further computation. However, the queries come unsorted for users, and there are queries regarding the same user with different movies. In this case, computing the similarity for the same user for multiple times wastes time. So I employ a dictionary here to group all the queries by user. Plus, I have to use a list of tuple to record the queries itself to maintain the order so that when I write the prediction to file in the original order. On the other hand, model-based CF derives a model for pairwise similarity for further computation, so I just maintain a tuple list to record the queries. Given the different stories about reading the dev.csv, the writing stories adapt respetively. When reading the data of ratings (train.csv), I employ sparse matrix(coo_matrix) for fast composing the user-movie-rating matrix. However, I have to use csr_matrix so I can index and slice the matrix faster. When I experiment with csr_matrix, I find it interering that for memory-based CF, csr_matrix is faster while for model_based CF, dense matrix seems better. So I use the faster matrix representation respectively. 

I use sklearn.preprocess.normalize to normalize the vector for computing cosine similarity, as the scipy.spatial.distance.cdict would fall into nan value when the norm of a vector is 0. First normalizing and then doing dot product would avoid this annoying situation.

For memory-based CF and model-based CF, the first one does the job in online fashion, so similarity is computer bewteen query and matrix while the second does the computation of simialrity pairwisely. As the time complexity for these two similatiry computation is the same, O(mn), the memory-based CF would cost more time as it repeat a lot of times for the computation. The functions in model.py and memory.py are self-explaining as I really implement the functions in layers. They do the same routine: computing similarity -> selecting knn -> predicting based on knnweight if needed. memoryCF/modelCF functions manage the work if not using pcc-like method for bias elimination and pccMemoryCF/pccModelCF manage the work if using pcc-like methods.

For bipartite clustering, BiCluster.py contains everything needed for it, and all come from the previous work. BiCF.py contains 2 functions(bi_item and bi_user) to do the real job that performing CF, based on k-nearest-centroids of bipartite clustering results. getCentroid function finds which cluster the query data belongs to. I actually used model-based CF here both for user similairty and item similarity as it is faster than memory-based CF(I discussed the reason in the above paragraph). After finding the query's centroids, instead of finding k-nearest-neighborhoods of it, finding k-nearest-centroids is used here. The work flow and logic is the same as before, just replacing the matrix and getting the centroids for queries.

In main.py, the main function parse the args and run the program accordingly. getBicluster.py is a very useful fucntion, it does the job of bipartite clustering and writes all the results, mapping realtions and centers values, into a file. So that I don't have to perform the bipartite job every time I call the CF function. I could just read the needed files and do CF without taking a long long time for waiting the results of bipartite clustering.


```
readHelper.py
	--readTrainModel(into dense matrix)
	--readTrainMemory(into sparse matrix like csr_matrix)
	--readQueryModel(into tuple list)
	--readQueryMemory(into dictionary and tuple list)

writeHelper.py
	--wirtePredModel.py(use tuple list)
	--writePredMemory.py(use dictionay and tuple list)
```
```
model.py					memory.py
	--similarity				--similarity
	--knn						--knn
	--knnweight					--knnweight
	--predict					--predict
	--memoryCF					--modelCF
	--pccMemoryCF				--PccModelCF
	--pccI						--pccI
	--pccU						--pccU
```
```
BiCluster.py  				BiCF.py
	--BiClustering				--bi_item
	--Kmean						--bi_user
	--getCluster				--getCentroid
	--getCenter
```


library used: numpy, scipy and sklean.preprocess.normalize

####4.3 Strengths and weaknesses of your design, and any problems that your system encountered.

I got into a situation that when it comes to row computation(numpy array operation with axis = 0), my program runs fast, while when it comes to column computation(numpy array operation with axis = 1), my program runs slowly. I realized that the default matrix storage for python/numpy is row-major matrix, C style. I could speed up for column computation if I use column-major matrix, Fortran style. That modification really helps me achieve equal fast speed for doing computation on both axes.

My program is really fast when k(the parameter for k-nearest-neighborhood) is small. But when k grows, the time spent increases noticably. It's not to hard to find this problem reading the running time from my result tables. It's definitely true that the running time would increase as k grows. But I have to be honest that for my implementation, the time grows so noticably. I use cprofile to check the running time of each modules and I find that these parts are relevantly time-consuming: slicing the sparse matrix, numpy.mean/numpy.average(used for getting predictions for mean or weight sum), numpy.argpartition(used for getting knn). I actually changed numpy.argsort into numpy.argpartition to get a little faster. Yet I believe there still are possibilities to optimize my implementation. I can do it if I have more time budget.(Yes,I can!!)

###5 How to run my program
```
Ariel:Collaborative-Filtering apple$ python main.py -h

usage: main.py [-h] [-t] [-m {memory,model}] [-k {10,100,500}]
               [-s {dotp,cosine}] [-w {mean,weight}] [-p] [-b]

collaborative filtering

optional arguments:
  -h, --help         show this help message and exit
  -t                 if running test
  -m {memory,model}  memory-based or model collaborative filtering
  -k {10,100,500}    number of k nearest neighborhood
  -s {dotp,cosine}   similarity metric used for knn
  -w {mean,weight}   approach for combining prediction given knn
  -p                 if standardization used
  -b                 if bipartite clustering userd

```

Please do not try -p and -b together as I don't have a part for doing bipartite clustering and pcc-like bias elimination together. It could be only called seperately.

Please do not try -t with other parameters, running test would only use the parameter setting that I hard coded, which is the best approach from my own observation.

This instructions will use the hard code setting to get prediction.txt for test.csv.


```
python main.py -t
```

I use memory-based user-user similarity, k = 100, dot product, mean, pcc-like method on movie bias, no bipartite clustering.

This instruction is a common use, without pcc-like method, without bipartite clustering:

```
python main.py -m model -k 100 -s cosine -w weight
```

This instruction is a common use with pcc-like method, without bipartite clustering:

```
python main.py -m model -k 100 -s cosine -w weight -p
```
This instruction is a common use without pcc-like method, with bipartite clustering:

```
python main.py -m model -k 100 -s cosine -w weight -b
```


