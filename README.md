# ANETS
Efficiently summarizing attributed diffusion networks
==========================================================================

Authors: Sorour E. Amiri, Liangzhe Chen, and B. Aditya Prakash

Date: September, 2018

Citing
------
If you find ANets useful in your research, we ask that you cite the following paper:
```
@Article{Amiri2018,
author="Amiri, Sorour E.
and Chen, Liangzhe
and Prakash, B. Aditya",
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
Note: You need to set the correct MATLAB_path in the makefile (Including the MATLAB executable).
```
- Example:
    MATLAB_path = '/Applications/MATLAB_R2016a.app/bin/matlab'
```    
To run DASSA do as follows,
```
>> make demo  
```


```
- Example: 
    python Anets.py
```



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
1	1 2.1 1.4
2	1.2   4.1 5 
3	1.3   3   7.1
```

Output:
-------


