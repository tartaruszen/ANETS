# ANETS Efficiently summarizing attributed diffusion networks (DAMI/ECML-PKDD 2018)
==========================================================================

Authors: Sorour E. Amiri, Liangzhe Chen, and B. Aditya Prakash

Date: September, 2018

Citing
------
If you find ANets useful in your research, we ask that you cite the following paper:
```
@Article{Amiri2018,
author="Amiri, Sorour E. and Chen, Liangzhe and Prakash, B. Aditya",
title="Efficiently summarizing attributed diffusion networks",
journal="Data Mining and Knowledge Discovery",
year="2018",
month="Sep",
day="01",
volume="32",
number="5",
pages="1251--1274",
issn="1573-756X",
doi="10.1007/s10618-018-0572-z",
url="https://doi.org/10.1007/s10618-018-0572-z"
}
```

Usage:
-----
To run ANets do as follows,
```
>> make
>> make demo  
```
First do 'make' (to compile sources). Then 'make demo' will run the ANets for toy example. 

To run ANets on a graph:

```
- Example: 
    python Anets.py <data_path> <percent> <num thread 1> <num thread 2> 
```
- <data_path>: Directory of the dataset

- < percent>: The percentage of compression

- <num thread 1>: Number of processors to summaries graphs in parallel

- <num thread 2>: Number of processors to generate the segmentation graphs in parallel


Input: 
------
- edges.txt : It is a tab separated file and index of nodes starts from 1 and are consecutive. Here is an example graph and its representation:

```
 1-------2
 |       |
 |       |
 |       |
 3 ----- 5
  \     /
   \   /
    \ /
     4
```
- The graph.txt file is:
```
Source	Target
1	2
1	3
3	4
3	5
5	4
```

- features.txt: It is a tab separated file. It shows the attributes of each node in the graph
node	feature1   feature2 feature3

```
node	feature1   feature2 feature3
1	    1          2.1      1.4
2	    1.2        4.1      5 
3	    1.3        3        7.1
```

Output:
-------
- links.txt: link list of the summary graph. It shows a weighted link from a super-node of another one. Here is an example:

```
source	target	weight
1	3	5.1011e-06
3	4	0.0025537
4	3	8.42034e-10
```

- features.txt: It shows the attributes of each super-node in the summary graph.
- node_map.txt: It shows the nodes in each super-nodes. Here is an example:

```
1: 1    2
3: 3    5
4: 4
```

