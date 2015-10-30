# Collaborative-Filtering
collaborative filtering on Netflix Dataset

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



