---
layout: post
title: Machine Learning:Decision Tree.
description: >
  Explanaion of what Decision Tree is and How to build A classification Tree.
tags: [Machine Learning]
author: author1
canonical_url: http://redwanidpto.github.io/2017/11/29/Just-Another-Post/
---
### What is Decision Tree ?

Decision Tree is a $$Supervised \ learning \ algorithm$$ ( [Types of Machine Learning](https://towardsdatascience.com/types-of-machine-learning-algorithms-you-should-know-953a08248861) ) . As its name suggests, it helps a machine to make decisions or to be more specific it helps to make predictions based on training data. There are mainly $$two$$ types of Decision Tree,

1. Classification Tree
2. Regression  Tree

Before explaining what $$Classification$$ and $$Regression$$ is, I would like to introduce an example, lets consider an example of Stock Exchange. We all are familiar with Stock Exchange market, right ? If not then have a look [here](https://en.wikipedia.org/wiki/Stock_exchange). Let there be 500 hundreds companies who sells there share. Buying a share can give you two types of feedback, Profit or Loss. Now if you want to buy a share of a particular company, there might be two types of question to ask. $$First$$, Whether this share is going to give you a profit or loss? $$Second$$, What will be the price of this share in a particular time period.

If you ask the first question then you are thinking  about $$Classification$$ and if you ask the second one then you are thinking about $$Regression$$. Now if we define more precisely what Classification and Regression is,

#### Classification :
>In classification we try to predict in which targeted class our test object belongs to. ( Discrete  or Categorical values )

#### Regression :
>In regression we try to find the relationship between attributes in the given training data, and try to predict the continuous outcome. ( Continuous values )

In this post our $$main \ focus$$ will be building a  $$Classification \ Tree$$. Lets build a Decision Tree for movie preference ( Classification ).

There are three attributes in this table Genre,  Rating, and Release Year. To make the Rating and Release year categorical we used binary split. Movies with Rating less than $$6.0$$ falls into one category and greater than or equal $$6.0$$ fall into another category. Movies with Release Year less than $$2000$$ falls into one category and greater than or equal $$2000$$ fall into another category

And we have two class for preference, either we like it ( $$YES$$ ), or Dislike it ($$ NO$$ )

| Genre     | Rating  | Release Year  | Class  |
| :-------- | ------: | :-----------: |:------:|
| Romantic  | => 6.0  |  => 2000      |  NO    |
| Romantic  |  < 6.0  |   < 2000      |  NO    |
| Romantic  | => 6.0  |  => 2000      |  NO    |
| Action    | => 6.0  |  => 2000      |  YES   |
| Action    |  < 6.0  |  => 2000      |  YES   |
| Action    | => 6.0  |   < 2000      |  YES   |
| Action    |  < 6.0  |   < 2000      |  NO    |
| Horror    |  < 6.0  |  => 2000      |  YES   |
| Horror    | => 6.0  |  => 2000      |  YES   |
| Horror    |  < 6.0  |   < 2000      |  NO    |

From this table we can build our decision tree like this,


![Decision Tree](https://raw.githubusercontent.com/redwandipto/Images/master/PostImage/dTree.jpg)


There can be multiple valid decision tree for this data set. But which decision tree is the better ? This depends on which attribute we choose to build our decision tree at each level. Maybe its hard to see the difference between trees built by choosing different attributes at different level of the tree but in large examples it makes a huge difference.

Choosing the best attribute at each level of the decision tree is important. Because this is what will determine how accurately the decision tree is going $$classify$$ the test object. Now question arises `How to choose best attribute?` Attributes with homogeneous class distribution is preferred. The more homogeneous the attribute is the more preferable it is. There are several techniques to choose homogeneous attributes.

Most common of them are,

1. Gini Index
2. Information Gain
3. Variance Reduction

Here we will discuss $$Information \ Gain$$ only. If you are interested then you can have a look on [Gini Index and Variance Reduction](https://en.wikipedia.org/wiki/Decision_tree_learning) also.

Information Gain is based on [Claude Shannon's](https://en.wikipedia.org/wiki/Claude_Shannon) $$Information \ Theory$$ introduced in $$1948$$ through his paper [“A Mathematical Theory of Communication”](http://sites.google.com/site/parthochoudhury/aMToC_CShannon.pdf). We will calculate $$Information \  Gain$$ of a particular $$Attribute$$ or $$Node$$ of the tree using `Shannon Entropy `$$H$$ .

$$
\begin{aligned}
  H = - \displaystyle\sum_{i=1}^{N} {P_i} \times {  \log_2{ (P_i)} }
\end{aligned}
$$

where $${P_i}$$ is the probability of occurrence of the $$i^{th}$$ possible value of the source symbol. In simple words $$Entropy$$ means how much $$noise$$ the data has. For example if we have $$N$$ number of Labels $$( l_1, l_2, ....., l_N )$$ in our data set. $$K_1$$ examples has class label of $$l_1$$, $$K_2$$ examples has class label of $$l_2$$ and so on. In general $$K_i$$ examples has $$l_i$$ class labels.

$$So, K = K_1 + K_2 + K_3 + .... + K_N$$,

$$Here, P_1 = \frac{K_1} {K}, P_2 = \frac{K_2} { K }, ....., P_i = \frac{K_i}{ K}$$;

$$
\begin{aligned}
  H = - \displaystyle\sum_{i=1}^{N} {P_i} \times {  \log_2{ (P_i)} }
\end{aligned}
$$

Doesn't make any sense ? Okay Let's have a look on the below picture.

![Entropy](https://raw.githubusercontent.com/redwandipto/Images/master/PostImage/entropy.png)

The above picture shows three different buckets with Red and Green balls. The $$1^{st}$$ Bucket contains only Red Balls thats why its homogeneous and Entropy value is $$Zero$$. The $$2^{nd}$$ bucket contains three Red Balls and One Green Ball so this bucket got some noise, and entropy is approximately $$0.81125$$ and the $$3^{rd}$$ bucket contains equal number of Red and Green Balls which mean the entropy is equal to $$One$$. If you are not still convinced you can put the values in the formula and you will get the same result. So the conclusion is `the Homogeneous property is inversely proportional to the Entropy. `

For your convenience lets calculate the Entropy for $$2^{nd}$$ bucket,

$$Here$$,

$$R = Red \ Ball$$,
$$G = Green \ Ball$$,    

$$So, \ R = 3, \ G = 1$$;

$$P_R = \frac{3}{4}, \ P_G = \frac{1}{4}$$ .

$$
\begin{aligned}
  H = - {P_R} \times { \log_2{ (P_R)} } - {P_G\times{ \log_2{(P_G)} }}  
\end{aligned}
$$

$$
\begin{aligned}
  H = - {\frac{3}{4}} \times {  \log_2{ (\frac{3}{4})} } - {\frac{1}{4}}\times{ \log_2{(\frac{1}{4})} }  = 0.811278
\end{aligned}
$$

The Information Gain will be the difference between $$Parent \ node$$ and the $$weighted \  average$$ entropies of its $$Child \ nodes$$. For example we are trying to calculate $$Information \  Gain$$ for node $$N$$, and we split examples at node $$N$$ into $$3$$ $$subsets$$, let them be $$N_1, N_2, N_3$$ then Information Gain At node $$N$$ will be the Difference between Entropy of node $$N$$ and the $$weighted \ average$$ of  $$N_1, N_2 \  and \  N_3$$.

If we have a set $$X$$ which contains $$N$$ examples. Now if we choose attribute $$T$$, which has $$L$$  values. Based on the values of $$attribute$$ $$T$$, the training example will split into $$L$$ $$subsets (X_1, X_2, ..., X_L) $$. Each subset $$X_i$$ has $$K_i$$ examples. So the $$Information \ Gain \ (IG )$$ will be,

$$
\begin{aligned}
  IG(X, L) = H(X) - \displaystyle\sum_{i=1}^{L} {\frac{K_i}{K}  }\times { H(X_i) }
\end{aligned}
$$

$$H( X )$$ and $$H( X_i )$$ are the entropies of the  training examples at each set based on their  class labels.

We will chose the $$attribute$$ which has highest $$IG(Information Gain)$$ at each level while building decision tree. But In case of continuous data we need to $$discretize$$ the data. There are several ways of $$Discretization$$. We will use $$Binary \ Discretization$$ for a particular $$Threshold$$ . Test chosen at each $$node$$ will be based on both $$attribute$$ and $$threshold$$. Examples less than threshold will go in the left child and examples with greater than or equal will go to the right child.

Now the question is which threshold is the best ? To answer this question we need to $$evaluate$$ a range of thresholds. The threshold which gives the most $$IG$$, is the best threshold. How to choose thresholds, will vary from problem to problem .

So the summary is, While building the $$Decision \ Tree$$ ( when values of $$attributes$$ are $$continuous$$ ) we will calculate $$IG$$ for all possible combination of $$attributes$$ and $$thresholds$$ ( from our chosen $$thresholds$$ list ). And then select the $$attribute$$ with best $$threshold$$. We will do this procedure for each $$node$$, and recursively build the whole $$Decision \  Tree$$. This will eventually give us an $$Optimized \  Decision\  Tree$$.

We can build a $$Randomized \ Decision \ Tree$$ by choosing the $$attribute$$ randomly with the best $$threshold$$. Sometimes $$Forest$$ of Randomized Trees gives pretty Good Predictions. In that case we take the $$average$$ of $$Probability \ distribution$$ of classified leaf node to predict more accurately.

**Notes:**

There is a problem with thess kind of learning algorithms, which is $$Data \ Overfitting$$.

#### What is Data Overfitting ?
> Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the models ability to generalize. [Reference](https://machinelearningmastery.com/overfitting-and-underfitting-with-machine-learning-algorithms/)  

We can overcome this problem by introducing a new technique called $$pruning$$. Typically $$leaf \ nodes$$ contains very few examples, which are not reliable. The distribution of classes among those few example may depend on luck than on any pattern among training examples. So, we will $$prune$$ those leaf nodes, which are not reliable.  

Here is Link of my Implementation of Decision Tree in C++. [Source Code](https://en.wikipedia.org/wiki/Decision_tree_learning)

Thanks for reading this long post. 😜 There might be a lot of mistakes in this article whether grammatical or conceptual, make sure to Cross check all the informations I have given. This was just a **Personal Note** though.





**References:**

[https://en.wikipedia.org/wiki/Decision_tree_learning](https://en.wikipedia.org/wiki/Decision_tree_learning)

[https://medium.com/udacity/shannon-entropy-information-gain-and-picking-balls-from-buckets-5810d35d54b4](https://medium.com/udacity/shannon-entropy-information-gain-and-picking-balls-from-buckets-5810d35d54b4)

[http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf](http://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s06/www/DTs.pdf)

[http://dataaspirant.com/2014/09/27/classification-and-prediction/](http://dataaspirant.com/2014/09/27/classification-and-prediction/)
