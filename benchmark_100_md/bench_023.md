# ECPE-2D: Emotion-Cause Pair Extraction based on Joint Two-Dimensional Representation, Interaction and Prediction

```meta
corpus_id: 220046814
```

## Authors

- Zixiang Ding dingzixiang@njust.edu.cn 
School of Computer Science and Engineering
Nanjing University of Science and Technology
China
- Rui Xia rxia@njust.edu.cn 
School of Computer Science and Engineering
Nanjing University of Science and Technology
China
- Jianfei Yu jfyu@njust.edu.cn 
School of Computer Science and Engineering
Nanjing University of Science and Technology
China

## Abstract

In recent years, a new interesting task, called emotion-cause pair extraction (ECPE), has emerged in the area of text emotion analysis. It aims at extracting the potential pairs of emotions and their corresponding causes in a document. To solve this task, the existing research employed a two-step framework, which first extracts individual emotion set and cause set, and then pair the corresponding emotions and causes. However, such a pipeline of two steps contains some inherent flaws: 1) the modeling does not aim at extracting the final emotion-cause pair directly; 2) the errors from the first step will affect the performance of the second step. To address these shortcomings, in this paper we propose a new end-toend approach, called ECPE-Two-Dimensional (ECPE-2D), to represent the emotion-cause pairs by a 2D representation scheme. A 2D transformer module and two variants, windowconstrained and cross-road 2D transformers, are further proposed to model the interactions of different emotion-cause pairs. The 2D representation, interaction, and prediction are integrated into a joint framework. In addition to the advantages of joint modeling, the experimental results on the benchmark emotion cause corpus show that our approach improves the F1 score of the state-of-the-art from 61.28% to 68.89%.

## Introduction

Emotion cause extraction (ECE), as a sub-task of emotion analysis, aims at extracting the potential causes of certain emotion expressions in text. The ECE task was first proposed by  and defined as a word-level sequence labeling problem. Gui et al. (2016a) released a new corpus and re-formalized the ECE task as a clause-level extraction problem. Given an emotion annotation, * Corresponding author the goal of ECE is to predict for each clause in a document if the clause is an emotion cause. This framework has received much attention in the following studies in this direction. Although the ECE task was well defined, it has two problems: Firstly, the emotion must be annotated manually before cause extraction, which greatly limits its practical application; Secondly, the way to first annotate the emotion and then extract the causes ignores the fact that emotions and causes are mutually indicative. To address this problem, we have proposed a new task named emotion-cause pair extraction (ECPE), aiming to extract the potential pairs of emotions and their corresponding causes together in our previous work .
Specifically, ECPE is defined as a fine-grained emotion analysis task, where the goal is to extract a set of valid emotion-cause pairs, given a document consisting of multiple clauses as the input. Figure 1 (a) shows an example of the ECPE task. The input in this example is a document consisting of six clauses. Clause c4 contains a "happy" emotion and it has two corresponding causes: clause c2 ("a policeman visited the old man with the lost money"), and clause c3 ("told him that the thief was caught"). Clause c5 contains a "worried" emotion and the corresponding cause is clause c6 ("as he doesn't know how to keep so much money"). The final output is a set of valid emotion-cause pairs defined at clause level: {c4-c2, c4-c3, c5-c6}. We have also proposed a two-step approach (ECPE-2Steps) to address the ECPE task . ECPE-2Steps is a pipeline of two steps:
Step 1 extracts an emotion set and a cause set individually. For example in Figure 1 (a), the emotion set is {c4, c5} and the cause set is {c2, c3, c6}; Step 2 conducts emotion-cause pairing and filtering based on the outputs of Step 1. As shown in Figure 1 (a), it first gets the candidate emotion-cause pairs by applying a Cartesian product to the emotion set and All possible  Step 2 -Filtering c1: Yesterday morning, c2: a policeman visited the old man with the lost money, c3: and told him that the thief was caught. c4: The old man was very happy. c5: But he still feels worried, c6: as he doesn't know how to keep so much money.

## Emotion set: {c4, c5}

Cause set: {c2, c3, c6}
Step 2 -Pairing
Step 1 (a) ECPE-2Step  c1-c3 c1-c4 c1-c5  c1-c2  c1-c1  c1-c6   c2-c3 c2-c4 c2-c5  c2-c2  c2-c1  c2-c6   c3-c3 c3-c4 c3-c5  c3-c2  c3-c1 c3-c6 cause set, and then train an independent filter to remove the invalid pairs.
Although the ECPE-2Steps approach seems reasonable and performs well, it still has the following shortcomings: (1) as a pipeline of two separate steps, ECPE-2Steps requires two prediction steps to get the final emotion-cause pair. The training of the model is also not directly aimed at extracting the final emotion-cause pair. (2) The errors from
Step 1 will affect the performance of Step 2. For one thing, the upper bound of the recall in Step 2 is determined by the recall in Step 1, because Step 2 cannot produce emotion-cause pairs from the emotions or causes that were not extracted by Step 1; for another, if Step 1 predicts too many incorrect emotions or causes, the precision of Step 2 will be reduced.
To address these problems, in this work we propose a new end-to-end ECPE solution, called ECPE-Two-Dimensional (ECPE-2D), to represent the emotion-cause pairs by a 2D representation scheme, and integrate the emotion-cause pair representation, interaction and prediction into a joint framework. As shown in Figure 1 (b), firstly, we design a 2D representation scheme to represent the emotion-cause pairs in forms of a square matrix, where each item represents an emotion-cause pair. Secondly, a 2D Transformer framework and its two variants, window-constrained and cross-road 2D transformers, are further proposed to capture the interaction between different emotion-cause pairs. Finally, we extract the valid emotion-cause pairs based on the 2D representation by conducting a binary classification on each emotion-cause pair. These three parts are integrated into a unified framework and trained simultaneously.
We evaluate our ECPE-2D approach on the benchmark emotion cause corpus. The experimental results prove that ECPE-2D can obtain overwhelmingly better results than the state-of-the-art methods on the emotion-cause pair extraction task and two auxiliary tasks (emotion extraction and cause extraction).

## Approach

## Overall Architecture

Following our prior work , we formalize the emotion-cause pair extraction (ECPE) task as follows. The input is a document consisting of multiple clauses d = [c 1 , c 2 , · · · , c |d| ], the goal of ECPE is to extract a set of emotioncause pairs in d:
where c emo is an emotion clause and c cau is the corresponding cause clause. The overall architecture of the proposed method is shown in Figure 2. It consists of three parts: 1) 2D Emotion-Cause Pair Representation; 2) 2D Emotion-Cause Pair Interaction; 3) 2D Emotion-Cause Pair Prediction. Firstly, an individual emotion/cause encoding component is firstly employed to obtain the emotion-specific representation vectors and cause-specific representation vectors. A full pairing component is applied to pair the two representation vectors into a 2D representation matrix. Then a 2D transformer module is proposed to model the interactions between different emotioncause pairs. For each emotion-cause pair in the matrix, the updated representation is finally fed to a softmax layer to predict if the pair is valid or not. The three modules are integrated into a unified framework and trained simultaneously.  Figure 2: Overview of the proposed joint framework for emotion-cause pair extraction.

## 2D Emotion-Cause Pair Representation

## Individual Emotion/Cause Encoding

The purpose of the clause encoder layer is to generate an emotion-specific representation and a causespecific representation for each clause in a document. The input is a document contains multiple clauses: d = [c 1 , c 2 , · · · , c |d| ], and each clause also contains multiple words
A hierarchical neural network which contains two layers is employed to capture such a word-clausedocument structure. The lower layer consists of a set of word-level Bi-LSTM modules, each of which corresponds to one clause and accumulate the context information for each word of the clause. The hidden state of the j-th word in the i-th clause h i,j is obtained based on a bi-directional LSTM. An attention mechanism is then adopted to get the clause representation s i .
The upper layer is composed of two independent components, with the goal to generate an emotionspecific representation r emo i and a cause-specific representation r cau i for each clause, respectively. Both components take the clause representation (s 1 , s 2 , , s |d| ) as input and use two clause-level Bi-LSTMs to obtain r emo
It should be noted that the individual emotion/cause encoder here is a compatible module.
Other emotion/cause encoder such as Inter-CE, Inter-EC , and BERT (Devlin et al., 2019) can also be used. We will compare and discuss them in the experiments.

## Emotion-Cause Full Pairing

In contrast to the ECPE-2Steps approach  which only extract pairs from the individual emotion set and cause set, we consider all possible pairs of clauses in d as candidates. Assuming the length of the document is |d|, then all possible pairs form a matrix M of the shape |d| * |d|, where the rows and columns represent the index of the emotion clause and the cause clause in the document, respectively. c emo i -c cau j is the element in the i-th row and the j-th column of M and indicates the emotion-cause pair that consists of the i-th clause and the j-th clause, encoded as:
where r emo i andŷ emo i are emotion-specific representation and emotion prediction of the i-th clause c i , r cau j andŷ cau j are cause-specific representation and cause prediction of the j-th clause c j . rpe i,j is a relative position embedding vector of c j relative to c i .

## 2D Emotion-Cause Pair Interaction

In the previous section, we have obtained a 2D representation matrix consisting of all possible emotion-cause pairs. Each element of the matrix represents a specific emotion-cause pair.
Considering that a document of length |d| will generate |d| * |d| possible emotion-cause pairs, a- mong which only a very small number of pairs are positive samples. Using the independent pair representation for emotion-cause pair prediction will not take advantage of this global information. Therefore, we further designed a 2D transformer for the ECPE task to effectively achieve the interaction between emotion-cause pairs.

## Standard 2D Transformer

The standard 2D transformer (Vaswani et al., 2017) consists of a stack of N layers. Each layer consists of two sublayers: a multi-head 2D self-attention mechanism followed by a position-wise feed forward network.
Multi-head 2D Self-attention. The multi-head 2D self-attention mechanism first calculates the query vector q i,j , key vector k i,j and value vector v i,j for each pair c emo i -c cau j in the document d as :
where W Q ∈ R n×n , W K ∈ R n×n , W V ∈ R n×n are parameters for queries, keys and values respectively. For each pair c emo i -c cau j , a set of weights β i,j = {β i,j,1,1 , β i,j,1,2 , · · · , β i,j,|d|,|d|) } are learned:
Then the new feature representation of c emo i -c cau j is obtained by considering all the |d| * |d| pairs in
Position-wise Feed Forward Network. In addition to the attention sublayer, a position-wise feed forward network is applied to each pair separately and identically:
It should be noted that both of the above two sublayers use the residual connection followed by normalization layer at its output:
As has mentioned, the standard 2D transformer consists of a stack of N layers. Let l denotes the index of transformer layers. The output of the previous layer will be used as the input of the next layer:
Computational inefficiency. Since the outputs of the standard transformer are |d| * |d| elements, each element requires the calculation of |d| * |d| attention weights, and eventually (|d| * |d|) * (|d| * |d|) weights are needed to be calculated and temporarily stored. To alleviate the computational load, we furthermore propose two variants of the standard 2D Transformer in the following two subsections: 1) window-constrained 2D Transformer and 2) cross-road 2D Transformer, as shown in Figure 3.

## 2D transformer

Time complexity Space complexity Standard O(batch * |d| * |d| * n * (|d| * |d| + n)) O(batch * |d| * |d| * (|d| * |d| + n)) Window-constrained O(batch * |d| * w * n * (|d| * w + n)) O(batch * |d| * w * (|d| * w + n)) Cross-road O(batch * |d| * |d| * n * (|d| + n)) O(batch * |d| * |d| * (|d| + n)) Table 1: Comparison of three kinds of 2D transformer in resource consumption. batch indicates the batch size during training, |d| indicates the number of clauses in the document, n refers to the hidden state size, w is equal to 2 * window + 1, and window is the window size used in window-constrained 2D transformer.

## Window-constrained 2D Transformer

Considering that most of the cause clauses are around the emotion clauses, we propose the window-constrained 2D transformer, which is a standard 2D transformer while only takes c emo
The outputs of the window-constrained 2D transformer are |d| * (window * 2 + 1) elements, each element requires the calculation of |d| * (window * 2 + 1) attention weights, and eventually (|d| * (window * 2+1)) * (|d| * (window * 2+1)) weights are needed to be calculated and temporarily stored.
It should be noted that compared to the standard 2D transformer, the window-constrained transformer not only greatly reduces the resource requirements, but also alleviates the class imbalance problem to some extent since most of the pairs out of the windows are negative samples.

## Cross-road 2D Transformer

Since the feature representation of pairs in the same row or column tends to be closer, we believe that pairs in the same row and column with the current pair have a greater impact on the current pair. Therefore, we propose the cross-road 2D transformer, in which the multi-head 2D self-attention mechanism is replaced by the cross-road 2D selfattention, and the other parts remain the same.
In the cross-road 2D self-attention, we calculate a set of row-wise weights β row i,j = {β row i,j,1 , β row i,j,2 , · · · , β row i,j,|d|) } and a set of column-
Then the new feature representation of c emo i -c cau j is obtained by considering the pairs in the same row and column with it:
The outputs of the cross-road 2D transformer are |d| * |d| elements, each element requires the calculation of (|d| + |d|) attention weights, and eventually (|d| * |d|) * (|d| * 2) weights are needed to be calculated and temporarily stored.
In this way, the new representation of each pair c emo i -c cau j can encode the information on all the pairs in the same row and column. In addition, if the cross-road 2D transformer is performed twice or more, the feature representation of each pair can encode the global information on all the pairs in M, while standard 2D transformer requires much more resource to achieve this.
We show an example of attentions to be calculated for standard, window-constrained, and crossroad 2D transformer in Figure 4 (a), (b), and (c), respectively, and summarize their resource consumption in Table 1.

## 2D Emotion-Cause Pair Prediction

After a stack of N 2D transformer layers, we can get the final representation o
The loss of emotion-cause pair classification for a document d is:
where y pair i,j is the ground truth distribution of emotion-cause pair of c emo i -c cau j .
In order to get better emotion-specific representation and cause-specific representation, we introduce the auxiliary loss for emotion prediction and cause prediction:
where y emo i and y cau i are emotion and cause annotation of clause c i , respectively. The final loss of our model for a document d is a weighted sum of L pair and L aux with L2-regularization term as follows:
where λ 1 , λ 2 , λ 3 ∈ (0, 1) are weights, θ denotes all the parameters in this model.

## Experiments

## Dataset and Metrics

We evaluated our proposed model on an ECPE corpus from , which was constructed based on a Chinese emotion cause corpus (Gui et al., 2016a). The same as (Xia and Ding, 2019), we stochastically select 90% of the data as training data and the remaining 10% as testing data. In order to obtain statistically credible results, we repeat the experiments 20 times and report the average result. The precision, recall, and F1 score defined in  are used as the metrics for evaluation.
In addition, we also evaluated the performance of two sub-tasks: emotion extraction and cause extraction, using the precision, recall, and F1 score defined in (Gui et al., 2016a) as the metrics.

## Experimental Settings

We use word vectors provided by  that were pre-trained on a corpora from Chinese Weibo. The dimensions of word embedding and relative position embedding are set to 200 and 50, respectively. The number of hidden units in BiLSTM for all our models is set to 100. The dimension of the hidden states, query, key, and value in the transformer are all set to 30. The window size in the window-constrained 2D transformer is set to 3. All weight matrixes and bias are randomly initialized by a uniform distribution U(0.01, 0.01).
For training details, we use the stochastic gradient descent (SGD) algorithm and Adam update rule with shuffled minibatch. The batch size and learning rate are set to 32 and 0.005, respectively. As for regularization, dropout is applied for word embeddings and the dropout rate is set to 0.7. The weights λ 1 , λ 2 , λ 3 in formula 20 are set to 1, 1, 1e-5, respectively. The code has been made publicly available on Github 1 . Table 2 shows the experimental results of our models and baseline methods on the ECPE task as well as two subtasks (emotion extraction and cause extraction).

## Overall Performance

ECPE-2Steps is a set of two-step pipeline methods proposed in our prior work , which first perform individual emotion extraction and cause extraction via multi-task learning, and then conduct emotion-cause pairing and filtering. Specifically, there are three kinds of multitask learning settings: 1) Indep: It is an independent multi-task learning method, in which emotion extraction and cause extraction are independently modeled.
2) Inter-CE: It is an interactive multi-task learning method, in which the predictions of cause extraction are used to improve emotion extraction.
3) Inter-EC: It is another interactive multi-task learning method, in which the predictions of emotion extraction are used to enhance cause extraction.
ECPE-2D is a joint framework proposed in this paper, which integrates the 2D emotion-cause pair representation, interaction, and prediction in an  Table 2: Performance of our models and baseline models (Xia and Ding 2019) using precision, recall, and F1measure as metrics on the ECPE task as well as the two sub-tasks.
end-to-end fashion. We explored three individual emotion/cause encoding settings: Indep, Inter-CE and Inter-EC, and three emotion-cause pair interaction settings:
1) "-" indicates that we do not introduce emotioncause pair interaction;
2) "+WC" indicates that we use the windowconstrained 2D transformer for emotion-cause pair interaction;
3) "+CR" indicates that we use the cross-road 2D transformer for emotion-cause pair interaction;
Note that due to the limitations of GPU memory, we have not been able to perform experiments with Standard 2D Transformer. First of all, it can be seen that our proposed model ECPE-2D (Inter-EC+WC) performs better than ECPE-2Step on all metrics of all tasks, which proves the effectiveness of our method.
On the ECPE task, ECPE-2Steps (Inter-EC) performs best among all the previous methods. Compared with ECPE-2Steps (Indep), the improvement of ECPE-2Steps (Inter-EC) is mainly on the recall rate, while the precision score is slightly reduced. On the basis of ECPE-2Steps (Inter-EC), the recall rate of ECPE-2D (Inter-EC+CR) has been further greatly improved, and the precision score has also been slightly improved, which ultimately leads to better performance on the F1 score.
On the emotion extraction and cause extraction subtasks, ECPE-2Steps (Inter-CE) and ECPE-2Steps (Inter-EC) achieves significant improvements compared to ECPE-2Steps (Indep) on the former and latter subtask respectively by leveraging the interaction between emotion and cause. While our method ECPE-2D (Inter-EC+CR) outperforms the previous methods on both subtasks. We attribute the improvements to multi-task learning, as compared to the ECPE-2Steps (Inter-EC) model, ECPE-2D (Inter-EC+CR) additionally introduces the emotion-cause pair extraction task and trains the three tasks in a unified framework.
In addition, we also explored the effect of using BERT 2 (Devlin et al., 2019) as clause encoder in Inter-EC, which is denoted as Inter-EC (BERT). The experimental results in Table 2 show that the performance on all tasks can be further greatly improved (especially, the state-of-the-art F1 score on the ECPE task is improved from 61.28% to 68.89%) by adopting BERT as clause encoder.

## ECPE-2D vs. ECPE-2Steps

In order to verify the effect of our proposed joint framework ECPE-2D, we discard the emotioncause pair interaction module and compare ECPE-2D models with ECPE-2Step models based on the same individual encoding setting, the results are shown in Table 2.
By comparing ECPE-2D (Indep) with ECPE-2Step (Indep), we find that the performance of ECPE-2D (Indep) on all the metrics of all tasks (especially the ECPE task) are significantly improved. On the ECPE task, the performance of ECPE-2D (Indep) is even better than ECPE-2D (Inter-EC), which is the prior state-of-the-art model. On the two subtasks, the performance has also been improved. We attribute the improvements to multi-task learning, as compared to the ECPE-2Step (Indep) model, ECPE-2D (Indep) additionally introduces the emotion-cause pair extraction task.
By comparing ECPE-2D (Inter-CE) and ECPE-2D (Inter-EC) with their two-step pipeline versions (ECPE-2Step (Inter-CE) and ECPE-2Step (Inter-EC)), we can draw similar conclusions. All these results prove that the proposed joint framework ECPE-2D is superior to the two-step pipeline framework ECPE-2Step in solving the ECPE task.

## The Effectiveness of 2D Transformer

Comparing with the ECPE-2D (Indep) model, the ECPE-2D (Indep+WC/CR) models can achieve further improvement on the ECPE task, while the improvement on the two subtasks are not significant. Similar conclusions can be drawn when comparing ECPE-2D (Inter-CE) and ECPE-2D (Inter-CE+WC/CR) as well as ECPE-2D(Inter-EC) and ECPE-2D(Inter-EC+WC/CR). Particularly, compared to the strong baseline ECPE-2D (Inter-EC(BERT)), the performance can still be improved by introducing two kinds of 2D transformers. These results demonstrate that the windowconstrained and cross-road 2D transformer can effectively improve the performance on the ECPE task via encoding interactive information between pairs.
In addition, we found that for ECPE-2D (Indep/Inter-CE/Inter-EC/Inter-EC(BERT)), the improvements brought by the introduction of window-constrained and cross-road 2D transformer are similar. These results indicate that the two 2D transformers are comparable.

## The Effectiveness of Auxiliary Supervision

In order to explore the impact of the auxiliary supervision of two subtasks (emotion extraction and cause extraction) on the final performance of the ECPE task, we design the experiments in Table 3. "-AS" denotes the auxiliary supervision is removed (in practice, we set λ 2 in formula (20) to 0). Compared with ECPE-2D (Indep/Inter-CE/Inter-EC), we find that the F1 score of ECPE-2D (Indep/Inter-CE/Inter-EC)-AS on the ECPE task decreased by about 1.4%, 2.2%, and 2.6%, respectively, which indicates that the supervisions of emo-  tion extraction and cause extraction are important for the ECPE task. Nevertheless, the results of ECPE-2D (Indep)-AS are still better than ECPE-2Step (Indep) and comparable to the prior stateof-the-art result, which shows that emotion-cause pair extraction can be performed individually and proves the effectiveness of our joint framework. Compared with ECPE-2D (Inter-EC+WC/+CR), the F1 score of ECPE-2D (Inter-EC+WC/+CR)-AS on the ECPE task decreased by about 1.1% and 0.8%, which is much less than the decrease between ECPE-2D (Inter-EC) and ECPE-2D (Inter-EC)-AS (drops 2.6%). These results lead to the conclusion that the negative impact of removing auxiliary supervision is reduced when pairwise encoders are introduced. From another perspective, when auxiliary supervisions are removed, the improvement brought by introducing pairwise encoders is greater. Comparing ECPE-2D (Inter-CE+WC/+CR), ECPE-2D (Indep+WC/+CR) and their "-AS" versions leads to similar conclusions. The above results again demonstrate the effectiveness of the proposed 2D transformer.

## Related Work

The emotion-cause pair extraction (ECPE) task was first proposed in our prior work  and is derived from the traditional emotion cause extraction (ECE) task. Since the ECPE task was recently proposed, there is little work on it. We mainly introduce the related work of ECE task.
The emotion cause extraction (ECE) task was first proposed by , with the goal to extract the word-level causes that lead to the given emotions in text. Based on the same task settings, there were some other individual studies that conducted ECE research on their own corpus us-ing rule-based methods (Neviarouskaya and Aono, 2013;Li and Xu, 2014;Gao et al., 2015a,b;Yada et al., 2017) or machine learning methods (Ghazi et al., 2015;Song and Meng, 2015).
Based on the analysis of the corpus in ,  suggested that a clause may be the most appropriate unit to detect causes and transformed the task from word-level to clauselevel. There was also some work based on this task setting (Russo et al., 2011;Gui et al., 2014). Recently, a Chinese emotion cause dataset was released by (Gui et al., 2016a,b;, and has received much attention. Based on this corpus, a lot of traditional machine learning methods (Gui et al., 2016a,b; and deep learning methods Li et al., 2018;Yu et al., 2019;Xu et al., 2019; were proposed.
In addition, there is also some work focused on cause detection for Chinese microblogs using a multiple-user structure and formalized two cause detection tasks for microblogs (current-subtweetbased cause detection and original-subtweet-based cause detection). (Cheng et al., 2017;Chen et al., 2018b,a).
The traditional ECE tasks suffer from two shortcomings: 1) the emotion must be annotated before cause extraction in ECE, which greatly limits its applications in real-world scenarios; 2) the way to first annotate emotion and then extract the cause ignores the fact that they are mutually indicative. To address this problem, we proposed the new emotion-cause pair extraction task in , which aims to extract the potential pairs of emotions and corresponding causes in a document. We have also proposed a two-step framework, which first extracts individual emotion set and cause set, and then pairs the corresponding emotions and causes. In this paper, we propose a new end-to-end approach to represent the emotion-cause pairs by a 2D representation scheme. Two kinds of 2D transformers, namely window-constrained and cross-road 2D transformers, are further proposed to model the interactions of different emotion-cause pairs. Finally, the 2D representation, interaction, and prediction are integrated into a joint framework.

## Conclusions

The emotion-cause pair extraction (ECPE) task has drawn attention recently. However the previous approach employed a two-step pipeline framework and has some inherent flaws. In this paper, instead of a pipeline of two steps, we propose a joint endto-end framework, called ECPE-2D, to represent the emotion-cause pairs by a 2D representation scheme, and integrate the 2D emotion-cause pair representation, interaction, and prediction into a joint a framework. We also develop two kinds of 2D Transformers, i.e., Window-constrained and Cross-road 2D Transformers, to further model the interaction of different emotion-cause pairs. The experimental results on the benchmark emotion cause corpus demonstrate that in addition to the advantages of joint modeling, our approach outperforms the state-of-the-art method by 7.6 percentage points in terms of the F1 score on the ECPE task.

## Figure 1 :

## Figure 3 :

## Figure 4 :

## Table 3 :

## Figures (text descriptions)

### Figure 1

An example showing two frameworks for solving the emotion-cause pair extraction (ECPE) task.

Figure 1 :
1An example showing two frameworks for solving the emotion-cause pair extraction (ECPE) task.

### Figure 2

Two simplified versions of 2D transformer for emotion-cause pair interaction.

Figure 3 :
3Two simplified versions of 2D transformer for emotion-cause pair interaction.

### Figure 3

Examples of attentions to be calculated in three 2D Transformers: (a) Standard 2D-Transformer, (b) Window-constrained 2D Transformer, and (c) Cross-road 2D Transformer.

Figure 4 :
4Examples of attentions to be calculated in three 2D Transformers: (a) Standard 2D-Transformer, (b) Window-constrained 2D Transformer, and (c) Cross-road 2D Transformer.

### Figure 4

predict the emotion-cause pair distributionŷ pair i,j as follows:

predict the emotion-cause pair distributionŷ pair i,j as follows:

### Figure 5

Performance of our models on the ECPE task when the auxiliary supervisions of emotion extraction and cause extraction are removed. For brevity, the prefix "ECPE-2D" of all methods in this table are omitted.

Table 3 :
3Performance of our models on the ECPE task when the auxiliary supervisions of emotion extraction and cause extraction are removed. For brevity, the prefix "ECPE-2D" of all methods in this table are omitted.

## References

1. Hierarchical convolution neural network for emotion cause detection on microblogs. Ying Chen, Wenjun Hou, Xiyao Cheng, International Conference on Artificial Neural Networks (ICANN). Ying Chen, Wenjun Hou, and Xiyao Cheng. 2018a. Hi- erarchical convolution neural network for emotion cause detection on microblogs. In International Conference on Artificial Neural Networks (ICANN), pages 115-122.

2. Joint learning for emotion classification and emotion cause detection. Ying Chen, Wenjun Hou, Xiyao Cheng, Shoushan Li, Empirical Methods in Natural Language Processing (EMNLP). Ying Chen, Wenjun Hou, Xiyao Cheng, and Shoushan Li. 2018b. Joint learning for emotion classification and emotion cause detection. In Empirical Method- s in Natural Language Processing (EMNLP), pages 646-651.

3. Emotion cause detection with linguistic constructions. Ying Chen, Sophia Yat Mei Lee, Shoushan Li, Chu-Ren Huang, Computational Linguistics (COLING). Ying Chen, Sophia Yat Mei Lee, Shoushan Li, and Chu- Ren Huang. 2010. Emotion cause detection with lin- guistic constructions. In Computational Linguistics (COLING), pages 179-187.

4. An emotion cause corpus for chinese microblogs with multiple-user structures. Xiyao Cheng, Ying Chen, Bixiao Cheng, Shoushan Li, Guodong Zhou, ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP). 171Xiyao Cheng, Ying Chen, Bixiao Cheng, Shoushan Li, and Guodong Zhou. 2017. An emotion cause corpus for chinese microblogs with multiple-user structures. ACM Transactions on Asian and Low-Resource Lan- guage Information Processing (TALLIP), 17(1):1- 19.

5. Bert: Pre-training of deep bidirectional transformers for language understanding. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT). Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understand- ing. In North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT), pages 4171-4186.

6. From independent prediction to re-ordered prediction: Integrating relative position and global label information to emotion cause identification. Zixiang Ding, Huihui He, Mengran Zhang, Rui Xia, AAAI Conference on Artificial Intelligence (AAAI). Zixiang Ding, Huihui He, Mengran Zhang, and Rui X- ia. 2019. From independent prediction to re-ordered prediction: Integrating relative position and global label information to emotion cause identification. In AAAI Conference on Artificial Intelligence (AAAI), pages 6343-6350.

7. Emotion cause detection for chinese micro-blogs based on ecocc model. Kai Gao, Hua Xu, Jiushuo Wang, Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD). Kai Gao, Hua Xu, and Jiushuo Wang. 2015a. Emotion cause detection for chinese micro-blogs based on e- cocc model. In Pacific-Asia Conference on Knowl- edge Discovery and Data Mining (PAKDD), pages 3-14.

8. A rulebased approach to emotion cause detection for chinese micro-blogs. Kai Gao, Hua Xu, Jiushuo Wang, Expert Systems with Applications. 429Kai Gao, Hua Xu, and Jiushuo Wang. 2015b. A rule- based approach to emotion cause detection for chi- nese micro-blogs. Expert Systems with Applications, 42(9):4517-4528.

9. Detecting emotion stimuli in emotion-bearing sentences. Diman Ghazi, Diana Inkpen, Stan Szpakowicz, International Conference on Intelligent Text Processing and Computational Linguistics (CICLing). Diman Ghazi, Diana Inkpen, and Stan Szpakowicz. 2015. Detecting emotion stimuli in emotion-bearing sentences. In International Conference on Intelli- gent Text Processing and Computational Linguistics (CICLing), pages 152-165.

10. A question answering approach to emotion cause extraction. Lin Gui, Jiannan Hu, Yulan He, Ruifeng Xu, Qin Lu, Jiachen Du, Empirical Methods in Natural Language Processing (EMNLP). Lin Gui, Jiannan Hu, Yulan He, Ruifeng Xu, Qin Lu, and Jiachen Du. 2017. A question answering ap- proach to emotion cause extraction. In Empirical Methods in Natural Language Processing (EMNLP), pages 1593-1602.

11. Event-driven emotion cause extraction with corpus construction. Lin Gui, Dongyin Wu, Ruifeng Xu, Qin Lu, Yu Zhou, Empirical Methods in Natural Language Processing (EMNLP). Lin Gui, Dongyin Wu, Ruifeng Xu, Qin Lu, and Yu Zhou. 2016a. Event-driven emotion cause extrac- tion with corpus construction. In Empirical Method- s in Natural Language Processing (EMNLP), pages 1639-1649.

12. Emotion cause extraction, a challenging task with corpus construction. Lin Gui, Ruifeng Xu, Qin Lu, Dongyin Wu, Yu Zhou, Chinese National Conference on Social Media Processing. Lin Gui, Ruifeng Xu, Qin Lu, Dongyin Wu, and Yu Zhou. 2016b. Emotion cause extraction, a chal- lenging task with corpus construction. In Chinese National Conference on Social Media Processing, pages 98-109.

13. Emotion cause detection with linguistic construction in chinese weibo text. Lin Gui, Li Yuan, Ruifeng Xu, Bin Liu, Qin Lu, Yu Zhou, Natural Language Processing and Chinese Computing (NLPCC). Lin Gui, Li Yuan, Ruifeng Xu, Bin Liu, Qin Lu, and Yu Zhou. 2014. Emotion cause detection with lin- guistic construction in chinese weibo text. In Nat- ural Language Processing and Chinese Computing (NLPCC), pages 457-464.

14. A text-driven rule-based system for emotion cause detection. Sophia Yat Mei Lee, Ying Chen, Chu-Ren Huang, NAACL HLT Workshop on Computational Approaches to Analysis and Generation of Emotion in Text. Sophia Yat Mei Lee, Ying Chen, and Chu-Ren Huang. 2010. A text-driven rule-based system for emotion cause detection. In NAACL HLT Workshop on Com- putational Approaches to Analysis and Generation of Emotion in Text, pages 45-53.

15. Text-based emotion classification using emotion cause extraction. Expert Systems with Applications. Weiyuan Li, Hua Xu, 41Weiyuan Li and Hua Xu. 2014. Text-based emotion classification using emotion cause extraction. Ex- pert Systems with Applications, 41(4):1742-1749.

16. A co-attention neural network model for emotion cause analysis with emotional context awareness. Xiangju Li, Kaisong Song, Shi Feng, Daling Wang, Yifei Zhang, Empirical Methods in Natural Language Processing (EMNLP). Xiangju Li, Kaisong Song, Shi Feng, Daling Wang, and Yifei Zhang. 2018. A co-attention neural network model for emotion cause analysis with emotional context awareness. In Empirical Methods in Natural Language Processing (EMNLP), pages 4752-4757.

17. Extracting causes of emotions from text. Alena Neviarouskaya, Masaki Aono, International Joint Conference on Natural Language Processing (IJCNLP). Alena Neviarouskaya and Masaki Aono. 2013. Extract- ing causes of emotions from text. In International Joint Conference on Natural Language Processing (IJCNLP), pages 932-936.

18. Emocause: an easy-adaptable approach to emotion cause contexts. Irene Russo, Tommaso Caselli, Francesco Rubino, Ester Boldrini, Patricio Martínez-Barco, Workshop on Computational Approaches to Subjectivity and Sentiment Analysis (WASSA). Irene Russo, Tommaso Caselli, Francesco Rubino, Es- ter Boldrini, and Patricio Martínez-Barco. 2011. E- mocause: an easy-adaptable approach to emotion cause contexts. In Workshop on Computational Approaches to Subjectivity and Sentiment Analysis (WASSA), pages 153-160.

19. Detecting concept-level emotion cause in microblogging. Shuangyong Song, Yao Meng, World Wide Web (WWW). Shuangyong Song and Yao Meng. 2015. Detecting concept-level emotion cause in microblogging. In World Wide Web (WWW), pages 119-120.

20. Attention is all you need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, Illia Polosukhin, Advances in Neural Information Processing Systems (NIPS). Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Pro- cessing Systems (NIPS), pages 5998-6008.

21. Emotion-cause pair extraction: A new task to emotion analysis in texts. Rui Xia, Zixiang Ding, Association for Computational Linguistics (ACL). Rui Xia and Zixiang Ding. 2019. Emotion-cause pair extraction: A new task to emotion analysis in texts. In Association for Computational Linguistics (ACL), pages 1003-1012.

22. RTHN: A RNN-transformer hierarchical network for emotion cause extraction. Rui Xia, Mengran Zhang, Zixiang Ding, International Joint Conference on Artificial Intelligence (IJCAI). Rui Xia, Mengran Zhang, and Zixiang Ding. 2019. RTHN: A RNN-transformer hierarchical network for emotion cause extraction. In International Joint Conference on Artificial Intelligence (IJCAI), pages 5285-5291.

23. Extracting emotion causes using learning to rank methods from an information retrieval perspective. Bo Xu, Hongfei Lin, Yuan Lin, Yufeng Diao, Liang Yang, Kan Xu, IEEE Access. 7Bo Xu, Hongfei Lin, Yuan Lin, Yufeng Diao, Liang Yang, and Kan Xu. 2019. Extracting emotion causes using learning to rank methods from an information retrieval perspective. IEEE Access, 7:15573-15583.

24. An ensemble approach for emotion cause detection with event extraction and multikernel svms. Ruifeng Xu, Jiannan Hu, Qin Lu, Dongyin Wu, Lin Gui, Tsinghua Science and Technology. 226Ruifeng Xu, Jiannan Hu, Qin Lu, Dongyin Wu, and Lin Gui. 2017. An ensemble approach for emo- tion cause detection with event extraction and multi- kernel svms. Tsinghua Science and Technology, 22(6):646-659.

25. A bootstrap method for automatic rule acquisition on emotion cause extraction. Shuntaro Yada, Kazushi Ikeda, Keiichiro Hoashi, Kyo Kageura, IEEE International Conference on Data Mining Workshops. Shuntaro Yada, Kazushi Ikeda, Keiichiro Hoashi, and Kyo Kageura. 2017. A bootstrap method for auto- matic rule acquisition on emotion cause extraction. In IEEE International Conference on Data Mining Workshops, pages 414-421.

26. Multiple level hierarchical network-based clause selection for emotion cause extraction. Xinyi Yu, Wenge Rong, Zhuo Zhang, Yuanxin Ouyang, Zhang Xiong, IEEE Access. 71Xinyi Yu, Wenge Rong, Zhuo Zhang, Yuanxin Ouyang, and Zhang Xiong. 2019. Multiple level hierarchical network-based clause selection for emotion cause extraction. IEEE Access, 7(1):9071-9079.
