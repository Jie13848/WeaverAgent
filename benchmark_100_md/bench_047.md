# OpenKE: An Open Toolkit for Knowledge Embedding

```meta
corpus_id: 53223679
```

## Authors

- Xu Han 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Institute for Artificial Intelligence
Tsinghua University
BeijingChina

State Key Lab on Intelligent Technology and Systems
Tsinghua University
BeijingChina
- Shulin Cao 
Institute for Artificial Intelligence
Tsinghua University
BeijingChina

State Key Lab on Intelligent Technology and Systems
Tsinghua University
BeijingChina

Knowledge Engineering Laboratory
Tsinghua University
BeijingChina

College of Information Science and Technology
Beijing Normal University
BeijingChina
- Xin Lv 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Knowledge Engineering Laboratory
Tsinghua University
BeijingChina
- Yankai Lin 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Institute for Artificial Intelligence
Tsinghua University
BeijingChina

State Key Lab on Intelligent Technology and Systems
Tsinghua University
BeijingChina
- Zhiyuan Liu 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Institute for Artificial Intelligence
Tsinghua University
BeijingChina

State Key Lab on Intelligent Technology and Systems
Tsinghua University
BeijingChina
- Maosong Sun 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Institute for Artificial Intelligence
Tsinghua University
BeijingChina

State Key Lab on Intelligent Technology and Systems
Tsinghua University
BeijingChina
- Juanzi Li 
Department of Computer Science and Technology
Tsinghua University
BeijingChina

Knowledge Engineering Laboratory
Tsinghua University
BeijingChina

## Abstract

We release an open toolkit for knowledge embedding (OpenKE), which provides a unified framework and various fundamental models to embed knowledge graphs into a continuous low-dimensional space. OpenKE prioritizes operational efficiency to support quick model validation and large-scale knowledge representation learning. Meanwhile, OpenKE maintains sufficient modularity and extensibility to easily incorporate new models into the framework. Besides the toolkit, the embeddings of some existing large-scale knowledge graphs pre-trained by OpenKE are also available, which can be directly applied for many applications including information retrieval, personalized recommendation and question answering. The toolkit, documentation, and pre-trained embeddings are all released on http://openke.thunlp.org/.

## Introduction

People construct various large-scale knowledge graphs (KGs) to organize structured knowledge about the world, such as WordNet (Miller, 1995), Freebase (Bollacker et al., 2008) and Wikidata (Vrandečić and Krötzsch, 2014). Most KGs are typically organized in the form of triples (h, r, t), with h and t indicating head and tail entities, and r indicating the relation between h and t, e.g., (Mark Twain, PlaceOfBirth, Florida). Abundant structured information in KGs is widely used to enhance various knowledge-driven NLP applications (e.g., information retrieval, question answering and dialogue system) with the ongoing effective construction of KGs.
Limited by the scale and sparsity of KGs, we have to represent KGs with corresponding distributed representations. Therefore, a variety of * indicates equal contribution † Corresponding author: Z. Liu(liuzy@tsinghua.edu.cn) knowledge embedding (KE) approaches have been proposed to embed both entities and relations in KGs into a continuous low-dimensional space, such as linear models (Bordes et al., 2011(Bordes et al., , 2014, latent factor models (Sutskever et al., 2009;Jenatton et al., 2012;Liu et al., 2017), neural models (Socher et al., 2013;Dong et al., 2014), matrix factorization models (Nickel et al., 2011(Nickel et al., , 2012(Nickel et al., , 2016Trouillon et al., 2016), and translation models (Bordes et al., 2013;Wang et al., 2014;Lin et al., 2015;Ji et al., 2015). These models have achieved great performance on benchmark datasets. However, there exist two main issues which may lead to difficulty in full utilization and further development. On the one hand, the existing implementations are scattered and unsystematic to some extent. For example, the interfaces of these model implementations are inconsistent with each other. On the other hand, these model implementations mainly focus on model validation and are often time-consuming, which makes it difficult to apply them for realworld applications. Hence, it becomes urgent to develop an efficient and effective open toolkit for KE, which will definitely benefit both the communities in academia and industry. For this purpose, we develop an open KE toolkit named "OpenKE". The toolkit provides a flexible framework and unified interfaces for developing KE models. While taking in some training and computing optimization methods, OpenKE makes KE models efficient and capable of embedding large-scale KGs. The features of OpenKE are threefold:
(1) At the data and memory level, the unified framework of OpenKE manages data and memory for KE models. Model developments based on OpenKE no longer require complicated data processing and memory allocation.
(2) At the algorithm level, OpenKE unifies the mathematical forms of various specific models to

## Model

Scoring Function Parameters Loss Function RESCAL (Nickel et al., 2011) h Mrt Mr ∈ R k×k , h ∈ R k , t ∈ R k margin-based loss TransE (Bordes et al., 2013)
TransH (Wang et al., 2014)
TransR (Lin et al., 2015) − Mrh + r − Mrt L1/L2 Mr ∈ R kr ×ke , r ∈ R kr , h ∈ R ke , t ∈ R ke margin-based loss TransD (Ji et al., 2015) − (rph p + I)h + r − (rpt p + I)t L1/L2 rp ∈ R kr , hp ∈ R ke , tp ∈ R ke , I ∈ R kr ×ke , r ∈ R kr , h ∈ R ke , t ∈ R ke margin-based loss
DistMult  < h, r, t > r ∈ R k , h ∈ R k , t ∈ R k logistic loss
HolE (Nickel et al., 2016)
ComplEx (Trouillon et al., 2016) (< h, r, t >) r ∈ C k , h ∈ C k , t ∈ C k logistic loss Table 1: The brief introduction of some typical KE models. For most models, k is the dimension of both entities and relations. For some other models, K e is the dimension of entities and k r is the dimension of relations. F denotes the Fourier transform. denotes the element-wise product. < a, b, c > denotes the element-wise multi-linear dot product.
implement them under the unified framework. We also propose a novel negative sampling strategy for further acceleration.
(3) At the computation level, OpenKE can separate a large-scale KG into several parts and adapt KE models for parallel training. Based on the underlying management of data and memory, we also adopt TensorFlow (Abadi et al., 2016) and Py-Torch (Paszke et al., 2017) to build a convenient platform to run models on GPUs.
Besides the toolkit, we also provide the pretrained embeddings of several well-known largescale KGs, which can be used directly for other relevant works without repeatedly spending much time for embedding KGs. In this paper, we mainly present the architecture design and implementation of OpenKE, as well as the benchmark evaluation results of some typical KE models implemented with OpenKE. Other related resources and details can be found on http://openke. thunlp.org/.

## Background

For a typical KG G, it expresses data as a directed graph G = {E, R, T }, where E, R and T indicate the sets of entities, relations and facts respectively. Each triple (h, r, t) ∈ T indicates there is a relation r ∈ R between h ∈ E and t ∈ E. For the entities h, t ∈ E and the relation r ∈ R, we use the bold face h, t, r to indicate their low-dimensional vectors respectively.
For any entity pair (h, t) ∈ E × E and any relation r ∈ R, we can determine whether there is a fact (h, r, t) ∈ T via their low-dimensional embeddings learned by KE models. These embeddings greatly facilitate understanding and mining knowledge in KGs. In practice, the KE models define a scoring function S(h, r, t) for each triple (h, r, t). In most cases, there are only true triples in KGs and non-existing triples can be either false or missing. Local closed world assumption (Dong et al., 2014) has been proposed to solve this problem, which requires existing triples to have higher scores than those non-existing ones. Hence, the scoring function S(h, r, t) returns a higher score if (h, r, t) is true, vice versa.
Based on the above-mentioned scoring functions, some KE models formalize a margin-based loss as the training objective to learn embeddings of the entities and relations:
Here [x] + indicates keeping the positive part of x and γ > 0 is a margin. T denotes the set of nonexisting triples, which is constructed by corrupting entities and relations in existing triples,
Some other KE models cast the training objective as a classification task. The embeddings of the entities and relations can be learned by minimizing the regularized logistic loss,
The main difference among various KE models is scoring functions. Hence, we briefly introduce several typical models and their scoring functions in Table 1. These models are state-of-the-art and widely introduced in many works. We systematically incorporate all of them into our OpenKE.

## Design Goals

Before introducing the concrete toolkit implementations, we report the design goals and features of OpenKE, including system encapsulation, operational efficiency, and model extensibility.

## Encapsulation

Developers tend to maximize the reuse of code to avoid unnecessary redundant development in practice. For KE, its task is fixed, and its experimental settings and model parameters are also similar. However, previous model implementations are scattered and lack of necessary interface encapsulation. Thus, developers have to spend extra time reading obscure open-source code and writing glue code for data processing when they construct models. In view of this issue, we build a unified underlying platform in OpenKE and encapsulate various data and memory processing which is independent of model implementations. As is shown in Figure 1, the system encapsulation makes it easy to train and test KE models. Thus, we just need to set hyperparameters via interfaces of the platform to construct KE models.

## Efficiency

Previous model implementations focus on model validation and enhancing experimental results rather than improving time and space efficiency. In fact, as real-world KGs can be very large, training efficiency is an important concern. Hence, OpenKE integrates efficient computing power, training methods, and various acceleration strategies to support KE models. We adopt TensorFlow and PyTorch to implement the model training and test modules based on the interfaces of underlying platform. These machine learning frameworks enable models to be run on GPU, with just few minutes needed for training and testing models on benchmark datasets. In order to train existing large-scale KGs, we also implement lightweight C/C++ versions for quick deployment and multithreading acceleration of KE models, in which some models (e.g. TransE) can embed more than 100M triples in a few hours on ordinary devices.

## Extensibility

Since different KE models have different design solutions, we make OpenKE fully extensible to future variants. For the underlying platform, we encapsulate data processing and memory manage-import config, Models, os os.environ['CUDA_VISIBLE_DEVICES']='0' con = config.Config() con.set_in_path('./FB15K/') con.set_work_threads(8) con.set_train_times(1000) con.set_alpha(0.001) con.set_margin(1.0) con.set_dimension(100) con.set_opt_method('SGD') con.init() con.set_model(models.TransE) con.run() 1, keep_dims = True) self.loss = tf.reduce_sum(tf.maximum(p_score -n_score + config.margin, 0)) ... ment, and then provide various data sampling interfaces. For the training modules, we provide enough interfaces for possible training methods. For the construction of KE models, we unify their mathematical forms and encapsulate them into a base class. These framework designs can greatly meet the needs of current and future models, and customized interfaces to meet individual requirements are also available in OpenKE. As shown in Figure 2, all specific models are implemented by inheriting the base class with designing their own scoring functions and loss functions. In addition, models in OpenKE can be placed into the framework of TensorFlow and PyTorch to interact with other machine learning models.

## Embeddings

## Implementations

In this section, we mainly present the implementations of acceleration modules and special sampling algorithm in OpenKE. OpenKE has been available to the public on GitHub 1 and is opensource under the MIT license.

## Algorithm 1 Parallel Learning

Require: Entity and relation sets E and R, training triples T = {(h, r, t)}. 1: Initialize all model embeddings and parameters. 2: for i ← 1 to epoches do 3:
In each thread: 4:
for j ← 1 to batches/threads do 5:
Sample a positive triple (h, r, t) 6:
Sample a corrupted triple (h , r , t ) 7:
Compute the loss function L 8:
Update the gradient ∇L 9: end for 10: end for 11: Return all embeddings and parameters

## GPU Learning

GPUs are widely used in machine learning tasks to speed up model training in recent years. In order to accelerate KE models, we integrate GPU learning mechanisms into OpenKE. We build the GPU learning platform based on TensorFlow (branch master) and PyTorch (branch OpenKE-PyTorch). Both TensorFlow and PyTorch are machine learning libraries, providing effective hardware optimizations and abundant arithmetic operators for convenient model constructions, especially the stable environments for GPU learning. The autograd packages also bring additional convenience. TensorFlow and PyTorch enable us to coustruct models without manual back propagation implementations, further reducing the programming complexity for GPU Learning. We develop necessary encapsulation modules aligning to Tensor-Flow and PyTorch so that the development and deployment of KE models can be faster and further convenient. Models can be deployed easily on a variety of devices without implementing complicated device setting code, even for multiple GPUs.

## Parallel Learning

Abundant computing resources (e.g Servers with multiple GPUs) do not exist all the time. In fact, we often rely on simple personal computers for model validation. Hence, we enable OpenKE to adapt models for parallel learning on CPU 2 besides employing GPU learning, which allow users to make full use of all available computing resources. The parallel learning method is shown in Algorithm 1. The main idea of parallel learning method is based on data parallelism mechanism, which divides training triples into several parts and trains each part of triples with a corresponding thread. In parallel learning, there are two strategies implemented to update gradients. One of the methods is the lock-free strategy, which means all threads share the unified embedding space and update embeddings directly without synchronized operations. We also implement a central synchronized method, where each thread calculates its own gradient and results will be updated after summing up the gradients from all threads.

## Offset-based Negative Sampling

All KE models learn their parameters by minimizing the margin-based loss function Eq. (1) or the regularized logistic loss Eq. (3). Both of these loss functions need to construct non-existing triples as negative samples. We have empirically found that the corrupted triples have great influence on final performance. Randomly replacing entities or relations with any other ones may make the negative triple set T contain some positive triples in T , which would weaken the performance of KE models. The original sampling algorithm will spend much time checking whether generated triples are in T and filtering them out. In OpenKE, we propose an offset-based negative sampling algorithm to generate negative triples. As shown in Figure 3, we renumber all entities with new serial numbers. Each entity's new number is obtained by adding an offset to its original ID, and the offset is the total number of positive entities which have lower IDs. Our algorithm first randomly sample a new number and then map the new number back to its corresponding entity. This algorithm can directly generate negative triples without any checking. Since the relation set is very small, we still directly replace positive relations for relation corruption.

## Evaluations

Link prediction has been widely used for evaluating KE models, which needs to predict the tail entity when given a triple (h, r, ?) or predict the   head entity when given a triple (?, r, t). In order to evaluate OpenKE, we implement various KE models with OpenKE, and compare their performance with previous works on link prediction task. Some datasets are usually used as benchmarks for link prediction, such as FB15K and WN18. FB15K is the relatively dense subgraph of Freebase; WN18 is the subset of WordNet. These public datasets are available online 3 . Following previous works, We adopt them in our experiments. The statistics of FB15K and WN18 are listed in Table 2, including the number of entities, relations, and facts.
As mentioned above, OpenKE supports models with efficient learning on both CPU and GPU. For CPU, the benchmarks are run on an Intel(R) Core(TM) i7-6700K @ 3.70GHz, with 4 cores and 8 threads. For GPU, the models in both TensorFlow and PyTorch versions are trained by GeForce GTX 1070 (Pascal), with CUDA v.8.0 (driver 384.111) and cuDNN v.6.5. To compare with the previous works, we simply follow the parameter settings used before and traverse all training triples for 1000 rounds. Other detailed parameters and training strategies are shown in our source code. We show these results in Table 3 and  Table 4. In these tables, the difference between our implementations and the paper reported results are listed in the parentheses. To demonstrate the efficiency of OpenKE, we select TransE as a representative and implement it with both OpenKE and KB2E 4 , and then compare their training time. KB2E is a widely-used toolkit for KE models on GitHub. These results can be found in Table 5.   From the results in Table 3, Table 4 and Table  5, we observe that: (1) Models implemented with OpenKE have the comparable accuracies compared to the values reported in the original papers. These results are compatible with our expectations. For some models, their accuracies are slightly higher due to OpenKE. These results indicate our toolkit is effecive. (2) OpenKE significantly accelerates the training process of the models trained both on CPU and GPU. As compared to the model implemented with KB2E, all models in OpenKE achieve more than 10× speedup. These results show that our toolkit is efficient.
The evaluation results indicate that our toolkit significantly handles the time-consuming problem and can support existing models to learn largescale KGs. In fact, TransE based on OpenKE only spends about 18 hours training the whole Wikidata for 10000 rounds and gets stable embeddings. There are more than 40M entities and 100M facts in Wikidata. We also evaluate the embeddings learned on the whole Wikidata on the link prediction task. Because the whole Wikidata is quite huge, we emphasize link prediction of Wikidata more on ranking a set of candidate entities rather than requiring one best answer. Hence, we report the proportion of correct entities in top-N ranked entities (Hits@10, Hits@20, Hits@50 and Hits@100) in Table 6. To our best knowledge, this is the first time that adopting KE models to embed an existing large-scale KG. The results shown in Table 6 indicate that OpenKE enables models to effectively and efficiently embed large-scale KGs.

## Conclusion

We propose an efficient open toolkit OpenKE for knowledge embedding. OpenKE builds a unified underlying platform to organize data and memory. It also applies GPU learning and parallel learning to speed up training. We also unify mathematical forms for specific models and encapsulate them to maintain enough modularity and extensibility. Experimental results demonstrate that the models implemented by OpenKE are efficient and effective.
In the future, we will incorporate more knowledge embedding models and maintain the stable embeddings of some large-scale knowledge graphs.

## Figure 2 :

## Figure 3 :

## Table 2 :

## Table 3 :

## Table 4 :

## Table 5 :

## Table 6 :

## Figures (text descriptions)

### Figure 1

An example for implementing a KE model (TransE) via OpenKE.

Figure 2 :
2An example for implementing a KE model (TransE) via OpenKE.

### Figure 2

An example for the offset-based negative sampling algorithm.

Figure 3 :
3An example for the offset-based negative sampling algorithm.

### Figure 3

tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False),

tf.reduce_sum(tf.reduce_mean(_n_score, 1, keep_dims = False),Datasets 
Memory 

Underlying Management 

Model 
Parameters 

Training 
Strategies 

Model Settings 

Training 
Evaluation 

Traning and Evaluation 

Code 
OpenKE 

Figure 1: An example for training a KE model 
(TransE) via OpenKE. 

import numpy as np 
import tensorflow as tf 
from Model import * 
class TransE(Model): 
def _calc(self, h, t, r): 
return abs(h + r -t) 
def embedding_def(self): 
config = self.get_config() 
self.ent_embeddings = tf.get_variable('ent_embeddings', 
[config.entTotal, config.hidden_size]) 
self.rel_embeddings = tf.get_variable('rel_embeddings', 
[config.relTotal, config.hidden_size]) 
... 
def loss_def(self): 
config = self.get_config() 
pos_h, pos_t, pos_r = self.get_positive_instance(in_batch = True) 
neg_h, neg_t, neg_r = self.get_negative_instance(in_batch = True) 
p_h = tf.nn.embedding_lookup(self.ent_embeddings, pos_h) 
p_t = tf.nn.embedding_lookup(self.ent_embeddings, pos_t) 
p_r = tf.nn.embedding_lookup(self.rel_embeddings, pos_r) 
n_h = tf.nn.embedding_lookup(self.ent_embeddings, neg_h) 
n_t = tf.nn.embedding_lookup(self.ent_embeddings, neg_t) 
n_r = tf.nn.embedding_lookup(self.rel_embeddings, neg_r) 
_p_score = self._calc(p_h, p_t, p_r) 
_n_score = self._calc(n_h, n_t, n_r) 
p_score = tf.reduce_sum(tf.reduce_mean(_p_score, 1, keep_dims = False), 
1, keep_dims = True) 
n_score =

### Figure 4

Statistics of FB15K and WN18.

Table 2 :
2Statistics of FB15K and WN18.Datasets 
FB15K 
Models 
TF 
PT 
MT 

TransE 
75.6(+28.5) 75.4(+28.3) 74.3(+27.2) 
TransH 
72.8(+14.3) 72.7(+14.2) 74.8(+16.3) 
TransR 
74.9(+6.2) 
75.7(+7.0) 
75.6(+6.9) 
TransD 
74.3(+0.1) 
74.2(+0.0) 
75.2(+1.0) 
RESCAL 49.1(+5.0) 
57.2(+13.1) 
-
DistMult 
73.4(+15.7) 75.4(+17.4) 
-
HolE 
70.4(−3.5) 
-
-
ComplEx 72.3(−11.7) 80.5(−3.5) 
-

### Figure 5

Experimental results of link prediction on WN18 (%).

Table 3 :
3Experimental results of link prediction on 
FB15K (%).

### Figure 6

Training time of different implementations of TransE on FB15K.

Table 4 :
4Experimental results of link prediction on WN18 (%).Models 
Time (s) 

TransE (KB2E, CPU) 
7124 
TransE (OpenKE, CPU, 1-Thread) 
386 
TransE (OpenKE, CPU, 2-Thread ) 
206 
TransE (OpenKE, CPU, 4-Thread) 
118 
TransE (OpenKE, CPU, 8-Thread) 
76 
TransE (OpenKE, GPU, TensorFlow) 
178 
TransE (OpenKE, GPU, PyTorch) 
266

### Figure 7

Experimental results of link prediction on the whole Wikidata.

Table 5 :
5Training time of different implementations of TransE on FB15K.

## References

1. Tensorflow: A system for large-scale machine learning. Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Proceedings of OSDI. OSDIMartín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghe- mawat, Geoffrey Irving, Michael Isard, et al. 2016. Ten- sorflow: A system for large-scale machine learning. In Proceedings of OSDI.

2. Freebase: a collaboratively created graph database for structuring human knowledge. Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, Jamie Taylor, Proceedings of KDD. KDDKurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. 2008. Freebase: a collaboratively cre- ated graph database for structuring human knowledge. In Proceedings of KDD.

3. Joint learning of words and meaning representations for open-text semantic parsing. Antoine Bordes, Xavier Glorot, Jason Weston, Yoshua Bengio, Proceedings of AISTATS. AISTATSAntoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. 2012. Joint learning of words and meaning repre- sentations for open-text semantic parsing. In Proceedings of AISTATS.

4. A semantic matching energy function for learning with multi-relational data. Antoine Bordes, Xavier Glorot, Jason Weston, Yoshua Bengio, Proceedings of ML. MLAntoine Bordes, Xavier Glorot, Jason Weston, and Yoshua Bengio. 2014. A semantic matching energy function for learning with multi-relational data. Proceedings of ML.

5. Translating embeddings for modeling multi-relational data. Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, Oksana Yakhnenko, Proceedings of NIPS. NIPSAntoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Ja- son Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Pro- ceedings of NIPS.

6. Learning structured embeddings of knowledge bases. Antoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio, Proceedings of AAAI. AAAIAntoine Bordes, Jason Weston, Ronan Collobert, Yoshua Bengio, et al. 2011. Learning structured embeddings of knowledge bases. In Proceedings of AAAI.

7. Knowledge vault: A web-scale approach to probabilistic knowledge fusion. Xin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, Wei Zhang, Proceedings of KDD. KDDXin Dong, Evgeniy Gabrilovich, Geremy Heitz, Wilko Horn, Ni Lao, Kevin Murphy, Thomas Strohmann, Shaohua Sun, and Wei Zhang. 2014. Knowledge vault: A web-scale ap- proach to probabilistic knowledge fusion. In Proceedings of KDD.

8. A latent factor model for highly multi-relational data. Rodolphe Jenatton, Nicolas L Roux, Antoine Bordes, Guillaume R Obozinski, Proceedings of NIPS. NIPSRodolphe Jenatton, Nicolas L Roux, Antoine Bordes, and Guillaume R Obozinski. 2012. A latent factor model for highly multi-relational data. In Proceedings of NIPS.

9. Knowledge graph embedding via dynamic mapping matrix. Guoliang Ji, Shizhu He, Liheng Xu, Kang Liu, Jun Zhao, Proceedings of ACL. ACLGuoliang Ji, Shizhu He, Liheng Xu, Kang Liu, and Jun Zhao. 2015. Knowledge graph embedding via dynamic mapping matrix. In Proceedings of ACL.

10. Learning entity and relation embeddings for knowledge graph completion. Yankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, Xuan Zhu, Proceedings of AAAI. AAAIYankai Lin, Zhiyuan Liu, Maosong Sun, Yang Liu, and Xuan Zhu. 2015. Learning entity and relation embeddings for knowledge graph completion. In Proceedings of AAAI.

11. Analogical inference for multi-relational embeddings. Hanxiao Liu, Yuexin Wu, Yiming Yang, Proceedings of ICML. ICMLHanxiao Liu, Yuexin Wu, and Yiming Yang. 2017. Analog- ical inference for multi-relational embeddings. In Pro- ceedings of ICML.

12. Wordnet: a lexical database for english. A George, Miller, Communications of the ACM. George A Miller. 1995. Wordnet: a lexical database for en- glish. Communications of the ACM.

13. Holographic embeddings of knowledge graphs. Maximilian Nickel, Lorenzo Rosasco, Tomaso Poggio, Proceedings of AAAI. AAAIMaximilian Nickel, Lorenzo Rosasco, and Tomaso Poggio. 2016. Holographic embeddings of knowledge graphs. In Proceedings of AAAI.

14. A three-way model for collective learning on multirelational data. Maximilian Nickel, Hans-Peter Volker Tresp, Kriegel, Proceedings of ICML. ICMLMaximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. 2011. A three-way model for collective learning on multi- relational data. In Proceedings of ICML.

15. Factorizing yago: scalable machine learning for linked data. Maximilian Nickel, Hans-Peter Volker Tresp, Kriegel, Proceedings of WWW. WWWMaximilian Nickel, Volker Tresp, and Hans-Peter Kriegel. 2012. Factorizing yago: scalable machine learning for linked data. In Proceedings of WWW.

16. Adam Paszke, Soumith Chintala, Ronan Collobert, Koray Kavukcuoglu, Clement Farabet, Samy Bengio, Iain Melvin, Jason Weston, Johnny Mariethoz, Pytorch: Tensors and dynamic neural networks in python with strong gpu acceleration. Adam Paszke, Soumith Chintala, Ronan Collobert, Ko- ray Kavukcuoglu, Clement Farabet, Samy Bengio, Iain Melvin, Jason Weston, and Johnny Mariethoz. 2017. Py- torch: Tensors and dynamic neural networks in python with strong gpu acceleration.

17. Reasoning with neural tensor networks for knowledge base completion. Richard Socher, Danqi Chen, D Christopher, Andrew Manning, Ng, Proceedings of NIPS. NIPSRichard Socher, Danqi Chen, Christopher D Manning, and Andrew Ng. 2013. Reasoning with neural tensor networks for knowledge base completion. In Proceedings of NIPS.

18. Modelling relational data using bayesian clustered tensor factorization. Ilya Sutskever, Joshua B Tenenbaum, Ruslan Salakhutdinov, Proceedings of NIPS. NIPSIlya Sutskever, Joshua B Tenenbaum, and Ruslan Salakhutdi- nov. 2009. Modelling relational data using bayesian clus- tered tensor factorization. In Proceedings of NIPS.

19. Complex embeddings for simple link prediction. Théo Trouillon, Johannes Welbl, Sebastian Riedel, Éric Gaussier, Guillaume Bouchard, Proceedings of ICML. ICMLThéo Trouillon, Johannes Welbl, Sebastian Riedel,Éric Gaussier, and Guillaume Bouchard. 2016. Complex em- beddings for simple link prediction. In Proceedings of ICML.

20. Wikidata: a free collaborative knowledgebase. Denny Vrandečić, Markus Krötzsch, Communications of the ACM. Denny Vrandečić and Markus Krötzsch. 2014. Wikidata: a free collaborative knowledgebase. Communications of the ACM.

21. Knowledge graph embedding by translating on hyperplanes. Zhen Wang, Jianwen Zhang, Jianlin Feng, Zheng Chen, Proceedings of AAAI. AAAIZhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph embedding by translating on hy- perplanes. In Proceedings of AAAI.

22. Embedding entities and relations for learning and inference in knowledge bases. Bishan Yang, Wen-Tau Yih, Xiaodong He, Jianfeng Gao, Li Deng, Proceedings of ICLR. ICLRBishan Yang, Wen-tau Yih, Xiaodong He, Jianfeng Gao, and Li Deng. 2015. Embedding entities and relations for learning and inference in knowledge bases. In Proceed- ings of ICLR.
