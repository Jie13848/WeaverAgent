# FedRec++: Lossless Federated Recommendation with Explicit Feedback

```meta
corpus_id: 235306107
```

## Authors

- Feng Liang liangfeng2018@email.szu.edu.cn 
National Engineering Laboratory for Big Data System Computing Technology
Shenzhen University
ShenzhenChina

College of Computer Science and Software Engineering
Shenzhen University
ShenzhenChina
- Weike Pan 
National Engineering Laboratory for Big Data System Computing Technology
Shenzhen University
ShenzhenChina

College of Computer Science and Software Engineering
Shenzhen University
ShenzhenChina
- Zhong Ming mingz@szu.edu.cn 
National Engineering Laboratory for Big Data System Computing Technology
Shenzhen University
ShenzhenChina

College of Computer Science and Software Engineering
Shenzhen University
ShenzhenChina

## Abstract

Privacy-aware recommendation with explicit feedback Input: A set of rating records for each user u, i.e., R u = {(u, i, r ui ); i ∈ I u }, where I u is a set of items rated by user u and r ui is the rating assigned to item i by user u.Goal: Estimate the preference of user u to the unrated items without exposing the rating scores and the rating behaviors of each user u (i.e., R u and I u ), which is the main difference between federated recommendation and traditional recommendation. Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 2 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 3 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 4 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 5 / 34 Related Work Probabilistic Matrix Factorization (PMF) In PMF [Mnih and Salakhutdinov, 2007], the rating of a user u to an item i is calculated via the inner product of their latent feature vectors, i.e.,r ui = U u· V T i· , where U u· , V i· ∈ R 1×d . Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 6 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 7 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021 8 / 34 Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI

## Introduction

Motivation Some very recent works propose to randomly sample some unrated items for each user and then assign some virtual ratings, so that the server can not identify the scores and the set of rated items easily during the server-client interactions. However, the virtual ratings assigned to the randomly sampled items will inevitably introduce some noise to the model training process, which will then cause loss in recommendation performance.

## Introduction

Notations (1/2) rating range r ui ∈ R rating of user u to item i R = {(u, i, r ui )} rating records in training data Ru rating records w.r.t. user u in R R te = {(u, i, r ui )} rating records in test data I the whole set of items Iu items rated by user u I u , |I u | = ρ|Iu| sampled unrated items w.r.t. user u from I\Iu U the whole set of users U i users who rated item i U i users who virtually rated item i U ũ i users who virtually rated item i w.r.t. denoiserũ U = {ũ} denoising clients (denoisers) y ui ∈ {0, 1} indicator variable Introduction Notations (2/2) In FedRec [Lin et al., 2020], in order to protect a user's rating behaviors, i.e., the set of items Iu rated by a user u, the authors design an effective hybrid filling (HF) strategy to randomly sample some unrated items. Firstly, it randomly samples |I u | unrated items of user u from I\Iu, where |I u | = ρ|Iu| with ρ ∈ {1, 2, 3}. Secondly, it uses the average rating or predicted rating of user u to a sampled item i as a virtual rating r ui . Thirdly, it calculates the gradients of user u to the rated items and the unrated items, i.e., ∇V i· , i ∈ Iu ∪ I u , and then uploads these gradients to the server. In this way, FedRec with the HF strategy achieves the purpose of protecting the user's original rating records and the rating behaviors in the preference modeling process. In particular, the virtual rating r ui is as follows,
where t denotes the number of iterations that have been executed in model training, and T predict is a parameter that determines when to start using the predicted rating as a virtual rating to a sampled rated item i.

## Federated Recommendation with Explicit Feedback (FedRec) (2/2)

In FedRec with PMF [Mnih and Salakhutdinov, 2007] as the backbone model, the gradient of each item i is as follows,
Notice that U i ∪ U i denotes the users that have rated or virtually rated item i, and
where r ui and r ui are the true observed rating and the virtual rating of user u to item i, respectively.
Although FedRec achieves privacy protection in rating prediction, the randomly sampled items in the hybrid filling strategy introduces some noise to the recommendation model, which inevitably affects the performance. This motivates us to design a lossless version of FedRec, which is critical to be deployed in a real-world application.
Secure Distributed Collaborative Filtering (SDCF) SDCF  uses stochastic gradient langevin dynamics (SGLD) [Welling and Teh, 2011] as a gradient descent method so as to defend differential attacks and prevent users' latent factors being leaked to the server. However, there may still be the leakage of the users' rating behaviors. For this reason, SDCF uses a two-stage randomized response algorithm to perturb the rated items and unrated items of each user, and then uploads the corresponding items' gradients to the server. Finally, SDCF can thus protect the users' rating behaviors similar to that of FedRec by uploading the virtually rated items' gradients.
As another issue, users have no ratings for the unrated items, hence the users can not calculate the values of the loss in the unrated items' gradients via r ui − U u· V T i· (i.e., e ui ), i ∈ I\I u . To solve this problem, SDCF samples some virtual e ui , i ∈ I\I u from the distribution of e ui , i ∈ I u of the user u. However, this strategy will also introduce some noise to the gradient at each iteration of model training.
Decentralized matrix factorization (DMF) [Chen et al., 2018] is a distributed POI recommendation framework for protecting users' rating data and solving the problem of computation and storage in the server. DMF keeps users' rating data in the corresponding local clients and utilizes these rating data to calculate the global items' latent factors and the local items' latent factors. And then each client synchronously sends the global items' gradients to his/her neighboring clients who are chosen by a random walk method.
Although this framework saves the resource of the server and avoids the risk of the rating data of all the users being leaked from the server to malicious attackers, there still exists the leakage of users' rating behaviors. Specifically, each user will receive the items' gradients from their own neighbors at each iteration of model training, and these items' gradients contain the items' IDs. Hence, this user will know the rated items of its neighbors, i.e., rating behaviors of its neighbors.
PDMFRec [Duriakova et al., 2019] is also a decentralized distributed POI recommendation framework with a novel, user-centric and privacy-enhanced matrix factorization method. This framework builds a user's adjacency graph in trustworthy clients via co-rated items between users, which solves the privacy problem of user geographic location leaked by DMF [Chen et al., 2018] when constructing the user adjacency graph. Furthermore, PDMFRec also proposes two privacy-protection settings that allow users to control the privacy-protection level. In the first setting, it allows each user to only choose parts of the rated items to take part in the construction of the user's adjacency graph. In the second setting, the rated items hidden by each user in the first setting also do not take part in model training. Compared with DMF, PDMFRec has comparable performance, better privacy-protection level, i.e., protecting the users' rating behaviors and ensuring the anonymity of the sending client. The anonymity of the sending client in PDMFRec inspires us to design an effective noise elimination strategy to prevent the denoising clients from colluding with the server, which will be described in detail later.

## Method

Our Solution: FedRec++ Figure: Illustration of the interactions between the server and each client in our loss federated recommendation (FedRec++). Notice that FedRec [Lin et al., 2020] is a special case of our FedRec++ without the denoising clients. Liang In the beginning, the server initializes the model parameters V i· , i ∈ I, and sends them to each client, i.e., step 1 in Figure 1. When each ordinary client u ∈ U\Ũ completes the computation of the item gradients, i.e., ∇V HF EF (u, i), i ∈ Iu ∪ I u , by using the local rating data, the server will receive the item gradients from each ordinary client u ∈ U\Ũ, i.e., step 2 in Figure 1. Then the server can calculate the summation of the gradients of each item i ∈ I as follows,
where U\Ũ denotes the ordinary clients (excluding the denoising clients). Since ∇V HF EF (u, i) with i ∈ Iu ∪ I u contain the gradients of client u to unrated items I u , the server can not identify each client u's rated items Iu easily. Hence, the rating behaviors of each client u are protected. Notice that ∇V i· in Eq.(4) is not immediately divided by |U i ∪ U i | as that in FedRec [Lin et al., 2020] via Eq.(2), but is divided by the number of users that have rated item i, i.e., |U i |, after noise elimination for more accurate preference modeling. We will show the details in Eq.(5) and Eq.(6).

## Method

Eliminate the Gradient Noise in the Server (2/3) The gradient noise (i.e., ∇V HF EF (u, i), i ∈ I u ) from each ordinary client u ∈ U\Ũ will inevitably bias the modeling of the user's preferences, which will be more serious when a larger value of ρ is used [Lin et al., 2020]. In order to eliminate the gradient noise in ∇V i· in Eq.(4), we design a specific algorithm for noise elimination in the server. Firstly, the server randomly selects some denoising clientsŨ ⊂ U as denoisers, which will be used to collect the gradient noise, i.e., ∇V HF EF (u, i), i ∈ I u , u ∈ U\Ũ, from the ordinary clients. Secondly, each denoising clientũ sends the summation of the noisy gradients of the ordinary clients (i.e., ∇V S (ũ, i)) to the server in order to eliminate the noise in ∇V i· in Eq.(4), which corresponds to step 3 in Figure 1. Notice that the server will also receive the number of users who virtually rated item i, i.e., |U ũ i |, from each denoising clientũ ∈Ũ in step 3 in Figure 1. Thirdly, once the server has received ∇V S (ũ, i) from all the denoisersŨ, the server can eliminate the gradient noise in ∇V i· in Eq.(4) as follows,
where ∇V S (ũ, i) also contains the gradients for item i of the denoiserũ itself, i.e., ∇V HF EF (ũ, i), i ∈ Iũ. We will describe the details in Eq.(9).

## Method

Eliminate the Gradient Noise in the Server (3/3)
After the server has eliminated the gradient noise in ∇V i· , the server can then calculate the number of users who rated item i, i.e.,
where γ denotes the learning rate. We depict the whole process of eliminating the gradient noise in the server in Algorithm 1. Liang After each client u ∈ U has received the item-specific latent feature vectors V i· , i ∈ I from the server, i.e., step 1 in Figure 1, each client u ∈ U can use its own local rating data to calculate the gradient ∇Uu·,
where e ui = r ui −r ui . Moreover, we can calculate the gradients ∇V HF EF (u, i), i ∈ Iu ∪ I u via Eq.(3), which is the same as that in FedRec [Lin et al., 2020].
Notice that ∇Uu· is used to update Uu· locally in each client, and ∇V HF EF (u, i) are sent to the server to update V i· with the denoising information, i.e., step 2 in Figure 1.
Notice that if a client u is a denoising client, it only needs to calculate the gradients ∇V HF EF (u, i) with i ∈ Iu rather than the gradients ∇V HF EF (u, i) with i ∈ Iu ∪ I u , because the gradients ∇V HF EF (u, i) with i ∈ Iu would be mixed with the gradient noise of the ordinary clients before being sent to the server. Hence, the privacy of the denoising client u does not leak towards the server, i.e., the rating behaviors of client u are protected.

## Method

Eliminate the Gradient Noise in the Client (2/3) When each ordinary client u ∈ U \Ũ sends ∇V HF EF (u, i) with i ∈ Iu ∪ I u to the server, they need to send the gradient noise (i.e., ∇V N (u, i), i ∈ I u ) to a denoising clientũ ∈Ũ at the same time, i.e., from the ordinary clients to the denoising clients of step 2 in Figure 1. We have the gradients ∇V N (u, i) as follows,
where I u denotes the sampled unrated items w.r.t. user u. Notice that transferring information between clients is a prominent specialty of the decentralized distributed framework. In our FedRec++, we adopt this specialty by sending gradient noise to the denoising clients, which can help eliminate the gradient noise.
For each denoising clientũ ∈Ũ , it does not need to send ∇V HF EF (ũ, i), i ∈ Iũ to the server immediately, because they can send ∇V S (ũ, i) containing ∇V HF EF (ũ, i), i ∈ Iũ to the server after collecting the gradient noise from the ordinary clients. We have ∇V S (ũ, i) as follows,
where the first term denotes the summation of the gradient noise sent to the denoising clientũ. Notice that the server can then use Eq.(5) to completely remove the noise, and the resulting ∇V i· on the left side of Eq.(5) contains the pure gradient of the rated item by the corresponding ordinary and/or denoising clients (i.e., users).

## Method

Eliminate the Gradient Noise in the Client (3/3) Notice that each ordinary client u ∈ U\Ũ only sends its own gradient noise to one denoising client. And each denoising clientũ does not need to send its own gradient noise (i.e., ∇V N (ũ, i)) to other denoisers, because each denoiserũ does not need to send the gradient noise ∇V HF EF (ũ, i), i ∈ I ũ to the server and thus the server does not need to eliminate the gradient noise in ∇V HF EF (ũ, i),ũ ∈Ũ. When each denoiserũ calculates the summation of the gradient noise from the ordinary clients, they can also calculate the number of users who virtually rated item i w.r.t. the denoiserũ at the same time, i.e., |U ũ i |. After each denoiserũ ∈Ũ obtains ∇V S (ũ, i), they send ∇V S (ũ, i) and |U ũ i | to the server, i.e., step 3 in Figure 1. Notice that ∇V S (ũ, i) in Eq.(9) will not leak the privacy of the denoising clients towards the server, because ∇V S (ũ, i) contains the gradient noise information of the ordinary client u ∈ U\Ũ. We describe the whole process of eliminating the gradient noise in the client in Algorithm 2.

## 17:

Calculate the number of users who rated item i, i.e., Sample items I u from I\Iu with |I u | = ρ|Iu |.

## 3:

Assign Uu· to U u· , and then update U u· via U u· ← U u· − γ∇U u· in T lcoal iterations.

## 4:

Assign a virtual rating for item i, i ∈ I u via Eq.(1).

## 5:

Calculate the gradient ∇Uu· via Eq. (7) and then update Uu· via Uu· ← Uu· − γ∇Uu·.

## 6:

for i ∈ Iu ∪ I u do 7:
Calculate ∇V HF EF (u, i) via Eq.(3).

## 8: end for

## 9:

Upload ∇V HF EF (u, i) with i ∈ Iu ∪ I u to the server.

## 10:

if u is not a denoising client then

## 11:

Calculate ∇V N (u, i) with i ∈ I u via Eq.(8), and send ∇V N (u, i) with i ∈ I u to a denoiserũ ∈Ũ . Receive the gradient noise ∇V N (u, i) with i ∈ I u from user u ∈ U \Ũ .

## 17:

Calculate the summation of the gradient noise u→ũ ∇V N (u, i) and the number of users who virtually rated item i w.r.t. the denoiserũ, i.e., |U ũ i |.

## 18:

Send the sum of gradient noise ∇V S (ũ, i) and |U ũ i | to the server.

## 19: end if

## Discussions

The designed noise elimination strategy in the server and in the client are actually quite generic and can be applied to other recommendation methods. For example, as introduced before, SDCF  uses a two-stage random response algorithm to perturb the rated items Iu and the unrated items I u of each user, and then calculates the gradients ∇V SDCF i· with i ∈ I u of each user to the unrated items. The gradients ∇V SDCF i· with i ∈ I u can be rewritten as follows,
where γ is the learning rate, e ui with i ∈ I u are sampled from the distribution of e ui with i ∈ Iu, Λ is a diagonal matrix related to the Gamma distribution of the regularization term of V i· , and I is an identity matrix with appropriate dimension. We can see that the gradients ∇V SDCF i· with i ∈ I u are similar to the gradient noise ∇V N (u, i) with i ∈ I u in Eq.(8) in our FedRec++. Hence, we can eliminate the noise introduced by the two-stage random response algorithm in SDCF via the denoising strategy in our FedRec++.

## Method

## Privacy Analysis

Firstly, each user's original rating records are always kept locally in the client in the whole process, which ensures the security of the original data.
Secondly, our FedRec++ adopts a hybrid filling strategy [Lin et al., 2020] to assign a virtual rating to each randomly sampled unrated item, which protects the users' rating behaviors.
Thirdly, each ordinary client u transfers the gradients of the sampled unrated items (i.e., the gradient noise ∇V N (u, i), i ∈ I u ) to a denoising client, which again does not reveal the user's rating behaviors (i.e., Iu).
Fourthly, the denoising client cannot identify the source (i.e., the sender) of the gradient, because the gradient does not contain the sensitive information of user ID, which guarantees the anonymity of each ordinary client [Duriakova et al., 2019].
Finally, even if the server colludes with the denoising clients, the server cannot obtain the rating behaviors of a specific user according to the gradient noise of the ordinary clients as collected by the denoising clients because of the anonymity of the clients (i.e., the denoising clients do not know which ordinary client the gradient noise belongs to). Hence, the server cannot obtain a user u's rating behaviors Iu via comparing the item ID of ∇V HF EF (u, i), i ∈ Iu ∪ I u uploaded to the server by user u and ∇V N (u, i), i ∈ I u sent to a denoising client by user u.

## Datasets and Evaluation Metrics

Besides using the two datasets in FedRec [Lin et al., 2020], i.e., MovieLens 100K (ML100K) and MovieLens 1M (ML1M), we also include a subset from Netflix (NF5K5K). Specifically, ML100K contains 100, 000 ratings of 1, 682 movies from 943 users; ML1M contains 1, 000, 209 ratings of 3, 952 movies from 6, 040 users; and NF5K5K contains 7, 944, 473 ratings of 5, 000 most popular movies from 5, 000 most active users.
We use two commonly used evaluation metrics, i.e., RMSE and MAE, for performance evaluation. Notice that the losslessness of our denoising strategy is independent of the datasets and the evaluation metrics. Liang Parameter Settings
For parameter configurations, we mainly follow FedRec [Lin et al., 2020]. In particular, we fix the number of latent features d = 20 and the number of iterations T = 100. We search the best value of the learning rate γ ∈ {0.7, 0.8, . . . , 1.4}, and have γ = 0.8, γ = 0.8 and γ = 1.0 on ML100K, ML1M and NF5K5K, respectively. We search the best value of the tradeoff parameter on the regularization terms α ∈ {0.1, 0.01, 0.001}, and have α = 0.001 on all the three datasets. We use different values of the sampling parameter ρ ∈ {0, 1, 2, 3}. We choose the best value of the iteration number T predict for starting filling the sampled unrated items via Eq.(1) and the iteration number T local for locally training U u· both from {5, 10, 15}, and have (T predict , T local ) = (10, 10), (T predict , T local ) = (5, 15) and (T predict , T local ) = (5, 15) on ML100K with ρ = 1, ρ = 2 and ρ = 3, respectively; and have (T predict , T local ) = (10, 15), (T predict , T local ) = (10, 15) and (T predict , T local ) = (10, 15) on ML1M with ρ = 1, ρ = 2 and ρ = 3, respectively; and have (T predict , T local ) = (5, 10), (T predict , T local ) = (5, 10) and (T predict , T local ) = (5, 15) on NF5K5K with ρ = 1, ρ = 2 and ρ = 3, respectively. All the hyper parameters are searched according to the MAE performance on the first copy of each dataset.

## Baselines

In order to study the effectiveness of our FedRec++, in particular of the merit of losslessness, we compare our FedRec++ with the most closely related work, i.e., FedRec [Lin et al., 2020].
In both FedRec and our FedRec++, we use PMF [Mnih and Salakhutdinov, 2007] as the backbone model and the hybrid filling strategy for virtual ratings [Lin et al., 2020]. Liang

## Observations:

The performance of our FedRec++ with ρ ∈ {1, 2, 3} is almost the same with that of FedRec with ρ = 0 (i.e., without introducing gradient noise), which means that our FedRec++ is able to completely eliminate the noise introduced when assigning virtual ratings to the sampled unrated items (i.e., the denoising strategy is lossless). The results are very promising and clearly show that our FedRec++ ensures privacy in model training without sacrificing the recommendation accuracy.
The performance of FedRec decreases with a larger value of ρ for higher security, which is expected because the volume of noise is proportional to the value of ρ. On the contrary, the performance of our FedRec++ does not decrease accordingly, which means that it can well eliminate the noise regardless of a small or large value of ρ.

## Observations:

When c = 0.2n, the performance of FedRec decreases fast as ρ increases, and the overall performance of our FedRec++ is significantly better than that of FedRec on each corresponding value of ρ, which again shows the effectiveness of the noise elimination strategy in our FedRec++. Because there are fewer clients participating in model training when c = 0.2n, the model training is insufficient and the error is higher as expected. Hence, when c = 0.2n, the performance of both FedRec and our FedRec++ with ρ = 1, 2, 3 are worse than that in Table 3.
When c ∈ {0.6n, n}, the performance of FedRec on ML100K does not decrease much with the increased values of ρ, while its performance on ML1M decreases.
This means that we may choose to use the noise elimination strategy in our FedRec++ appropriately for different datasets. Importantly, the performance of our FedRec++ with ρ ∈ {1, 2, 3} is almost the same with that in Table 3, which means that we may only use 60% clients in model training to achieve comparable performance as that of using all the clients.
Results(5/6) Results(6/6)

## Observations:

Using more denoising clients (i.e., a larger value of η) can more efficiently process the gradient noise in parallel, which thus reduces the cost of each denoising client.
The cost of each ordinary client is almost the same when η ∈ {1, n/4, n/2}, which is expected since it is independent of the number of denoising clients.
When η ∈ {n/4, n/2}, the cost of each client is higher than that when η ∈ {0, 1}, because there are more denoising clients sending gradients to the server. Quantitatively, the cost of

## Conclusions

In this paper, we study an emerging problem, i.e., privacy-aware recommendation with explicit feedback. In particular, we propose a novel and lossless federated recommendation method called FedRec++, for which we use some denoising clients to completely eliminate the noise caused by the assigned virtual ratings to some randomly sampled items.
We also conduct privacy analysis to show that our FedRec++ is able to protect the user privacy well. Moreover, our FedRec++ is a generic solution, which embodies FedRec [Lin et al., 2020] as a special case, and the denoising strategy can also be used in the other privacy-aware recommendation method such as SDCF .
Experimental results on three public datasets show the effectiveness (i.e., losslessness) and efficiency (i.e., low communication cost) of our FedRec++. Liang We are interested in generalizing our FedRec++ to some models with ranking losses (e.g., pairwise loss [Rendle et al., 2009] or listwise loss [Wu et al., 2018]), neural network models, and some vertical federated machine learning settings [Yang et al., 2019, Zhang et al., 2020. We are also interested in federating some more advanced recommendation models such as those based on deep learning techniques [He et al., 2017, Liang et al., 2018, Sun et al., 2019. Moreover, we will further explore the applicability of the denoising strategy to other works.

## Table :

## Table :

## Table :

## Table :

## Figures (text descriptions)

### Figure 1

The algorithm of FedRec++ in the server.1: Randomly select some clients as denoisers, i.e.,Ũ . 2: Initialize the model parameters V i· , i = 1, 2, . . . , m and send them to each client u ∈ U . 3: for t = 1, 2, . . . , T do 4:for each client u ∈ U in parallel do5:ClientTraining(V i· , i = 1, 2, . . . , m; TRAINING; u;ũ;Ũ ; gradient ∇V i· via Eq.(4) and also |U i ∪ U i |. ). /*Wait for all the clients to complete.*/ 15: for i = 1, 2, . . . , m do16:Eliminate the gradient noise in ∇V i· via Eq.(5).

The algorithm of FedRec++ in the server.1: Randomly select some clients as denoisers, i.e.,Ũ . 2: Initialize the model parameters V i· , i = 1, 2, . . . , m and send them to each client u ∈ U . 3: for t = 1, 2, . . . , T do 4:for each client u ∈ U in parallel do5:ClientTraining(V i· , i = 1, 2, . . . , m; TRAINING; u;ũ;Ũ ; gradient ∇V i· via Eq.(4) and also |U i ∪ U i |. ). /*Wait for all the clients to complete.*/ 15: for i = 1, 2, . . . , m do16:Eliminate the gradient noise in ∇V i· via Eq.(5).

### Figure 2

each client is at most 350 − 170 = 180, 551 − 265 = 286, and 4608 − 2535 = 2073 more on ML100K, ML1M and NF5K5K, respectively. Because there are 100 iterations in model training, the averaged additional communication costs for each client are 180 × 80 × 100 bytes = 1.37 MB, 286 × 80 × 100 bytes = 2.18 MB, and 2073 × 80 × 100 bytes = 15.8 MB, on ML100K, ML1M and NF5K5K, respectively. We can see that the noise elimination strategy in our FedRec++ only consumes a small amount of communication cost of clients, which shows another merit of our FedRec++.

each client is at most 350 − 170 = 180, 551 − 265 = 286, and 4608 − 2535 = 2073 more on ML100K, ML1M and NF5K5K, respectively. Because there are 100 iterations in model training, the averaged additional communication costs for each client are 180 × 80 × 100 bytes = 1.37 MB, 286 × 80 × 100 bytes = 2.18 MB, and 2073 × 80 × 100 bytes = 15.8 MB, on ML100K, ML1M and NF5K5K, respectively. We can see that the noise elimination strategy in our FedRec++ only consumes a small amount of communication cost of clients, which shows another merit of our FedRec++.

### Figure 3

Some notations and their explanations.

Table :
:Some notations and their explanations.Notation 
Explanation 

n 
number of users (i.e., clients) 
m 
number of items 
R = {1, . . . , 5}

### Figure 4

Some notations and their explanations (cont.).

Table :
:Some notations and their explanations (cont.).Notation 
Explanation 

d ∈ R 
number of latent dimensions 
Uu·, U u· ∈ R 1×d 
user-specific latent feature vector 
V i· ∈ R 1×d 
item-specific latent feature vector 
r ui 
predicted rating of user u to item i 
γ 
learning rate 
ρ 
sampling parameter 
c 
number of clients in training 
η 
number of denoising clients 
λ 
tradeoff parameter 
T 
iteration number 
Federated Recommendation with Explicit Feedback 
(FedRec) (1/2)

### Figure 5

Figure:Recommendation performance of FedRec and our FedRec++ with different values of c ∈ {0.2n, 0.6n, n} and ρ ∈ {1, 2, 3}. Notice that we fix η = 1 and the results on NF5K5K are similar to that on ML100K.

Table :
:Recommendation performance of FedRec and our FedRec++ with different values of ρ ∈ {1, 2, 3}. Notice that we fix 
c = n and η = 1 in our FedRec++, and copy the results of FedRec on ML100K and ML1M from [Lin et al., 2020] for reference 
and direct comparison. 

Data 
Algorithm 
MAE 
RMSE 
ρ 

ML100K 

FedRec 
0.7418± 0.0048 
0.9424± 0.0064 
0 
FedRec 
0.7440± 0.0043 
0.9432± 0.0056 
1 
FedRec++ 
0.7417± 0.0049 
0.9422± 0.0063 
FedRec 
0.7445± 0.0045 
0.9431± 0.0057 
2 
FedRec++ 
0.7422± 0.0047 
0.9430± 0.0061 
FedRec 
0.7447± 0.0043 
0.9431± 0.0054 
3 
FedRec++ 
0.7416± 0.0049 
0.9421± 0.0064 

ML1M 

FedRec 
0.7193± 0.0012 
0.9106± 0.0015 
0 
FedRec 
0.7217± 0.0011 
0.9129± 0.0012 
1 
FedRec++ 
0.7198± 0.0011 
0.9113± 0.0013 
FedRec 
0.7239± 0.0011 
0.9152± 0.0011 
2 
FedRec++ 
0.7195± 0.0011 
0.9109± 0.0013 
FedRec 
0.7263± 0.0010 
0.9178± 0.0010 
3 
FedRec++ 
0.7196± 0.0013 
0.9109± 0.0015 

NF5K5K 

FedRec 
0.7139± 0.0007 
0.9090± 0.0008 
0 
FedRec 
0.7148± 0.0004 
0.9102± 0.0005 
1 
FedRec++ 
0.7137± 0.0008 
0.9088± 0.0012 
FedRec 
0.7152± 0.0005 
0.9104± 0.0005 
2 
FedRec++ 
0.7137± 0.0002 
0.9089± 0.0002 
FedRec 
0.7160± 0.0007 
0.9110± 0.0005 
3 
FedRec++ 
0.7138± 0.0005 
0.9090± 0.0004

### Figure 6

Average communication cost per iteration of each denoising client, each ordinary client and each (denoising or ordinary) client in our FedRec++ with different numbers of denoising clients η ∈ {0, 1, n/4, n/2}. Notice that we fix c = n and ρ = 1. The unit of each cost is 80 bytes occupied by one latent vector.

Figure:Recommendation performance of FedRec and our FedRec++ with different values of c ∈ {0.2n, 0.6n, n} and ρ ∈ {1, 2, 3}. Notice that we fix η = 1 and the results on NF5K5K are similar to that on ML100K.Liang, Pan and Ming (Shenzhen U.) 
FedRec++ 
AAAI 2021 
27 / 34 

Experiments 

Results(3/6) 

ML100K 
ML1M 

Liang, Pan and Ming (Shenzhen U.) 
FedRec++ 
AAAI 2021 
28 / 34 

Experiments 

Results(4/6)

### Figure 7

, Pan and Ming (Shenzhen U.)

Table :
:Average communication cost per iteration of each denoising client, each ordinary client and each (denoising or ordinary) client in our FedRec++ with different numbers of denoising clients η ∈ {0, 1, n/4, n/2}. Notice that we fix c = n and ρ = 1. The unit of each cost is 80 bytes occupied by one latent vector.Data 
Denoising 
Ordinary 
Client 
η 
Client 
Client 

ML100K 

0 
170 
170 
0 
81,550 
254 
258 
1 
571 
256 
350 
n/4 
251 
256 
294 
n/2 

ML1M 

0 
265 
265 
0 
804,058 
397 
399 
1 
903 
397 
551 
n/4 
393 
397 
460 
n/2 

NF5K5K 

0 
2535 
2535 
0 
6,324,227 
3799 
3801 
1 
7309 
3800 
4608 
n/4 
3537 
3799 
4172 
n/2 

Liang, Pan and Ming (Shenzhen U.) 
FedRec++ 
AAAI 2021 
30 / 34

## References

1. . C Chen, Z Liu, P Zhao, J Zhou, Li , X , Chen, C., Liu, Z., Zhao, P., Zhou, J., and Li, X. (2018).

2. Privacy preserving point-of-interest recommendation using decentralized matrix factorization. Proceedings of the 32nd AAAI Conference on Artificial Intelligence. the 32nd AAAI Conference on Artificial IntelligencePrivacy preserving point-of-interest recommendation using decentralized matrix factorization. In Proceedings of the 32nd AAAI Conference on Artificial Intelligence, pages 257-264.

3. Pdmfrec: A decentralised matrix factorisation with tunable user-centric privacy. E Duriakova, E Z Tragos, B Smyth, N Hurley, F J Peña, P Symeonidis, J Geraci, A Lawlor, Proceedings of the 13th ACM Conference on Recommender Systems. the 13th ACM Conference on Recommender SystemsDuriakova, E., Tragos, E. Z., Smyth, B., Hurley, N., Peña, F. J., Symeonidis, P., Geraci, J., and Lawlor, A. (2019). Pdmfrec: A decentralised matrix factorisation with tunable user-centric privacy. In Proceedings of the 13th ACM Conference on Recommender Systems, pages 457-461.

4. . X He, L Liao, H Zhang, L Nie, X Hu, T Chua, He, X., Liao, L., Zhang, H., Nie, L., Hu, X., and Chua, T. (2017).

5. Neural collaborative filtering. Proceedings of the 26th International Conference on World Wide Web. the 26th International Conference on World Wide WebNeural collaborative filtering. In Proceedings of the 26th International Conference on World Wide Web, pages 173-182.

6. . J Jiang, C Li, Lin , S , Jiang, J., Li, C., and Lin, S. (2019).

7. Towards a more reliable privacy-preserving recommender system. Information Sciences. 482Towards a more reliable privacy-preserving recommender system. Information Sciences, 482:248-265.

8. Variational autoencoders for collaborative filtering. D Liang, R G Krishnan, M D Hoffman, T Jebara, Proceedings of the 2018 World Wide Web Conference on World Wide Web. the 2018 World Wide Web Conference on World Wide WebLiang, D., Krishnan, R. G., Hoffman, M. D., and Jebara, T. (2018). Variational autoencoders for collaborative filtering. In Proceedings of the 2018 World Wide Web Conference on World Wide Web, pages 689-698.

9. . G Lin, F Liang, W Pan, Ming , Z , Lin, G., Liang, F., Pan, W., and Ming, Z. (2020).

10. Fedrec: Federated recommendation with explicit feedback. IEEE Intelligent SystemsFedrec: Federated recommendation with explicit feedback. IEEE Intelligent Systems.

11. . A Mnih, R R Salakhutdinov, Mnih, A. and Salakhutdinov, R. R. (2007).

12. Probabilistic matrix factorization. Probabilistic matrix factorization.

13. Proceedings of the 21st International Conference on Neural Information Processing Systems. the 21st International Conference on Neural Information Processing SystemsIn Proceedings of the 21st International Conference on Neural Information Processing Systems, pages 1257-1264.

14. BPR: Bayesian personalized ranking from implicit feedback. S Rendle, C Freudenthaler, Z Gantner, L Schmidt-Thieme, Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence. the 25th Conference on Uncertainty in Artificial IntelligenceRendle, S., Freudenthaler, C., Gantner, Z., and Schmidt-Thieme, L. (2009). BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence, pages 452-461.

15. . F Sun, J Liu, J Wu, C Pei, X Lin, W Ou, P Jiang, Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., and Jiang, P. (2019).

16. . Pan Liang, Ming , FedRec++ AAAI. 2021Liang, Pan and Ming (Shenzhen U.) FedRec++ AAAI 2021

17. References Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. Proceedings of the 28th ACM International Conference on Information and Knowledge Management. the 28th ACM International Conference on Information and Knowledge ManagementReferences Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 1441-1450.

18. . M Welling, Y W Teh, Welling, M. and Teh, Y. W. (2011).

19. Bayesian learning via stochastic gradient langevin dynamics. Proceedings of the 28th International Conference on Machine Learning. the 28th International Conference on Machine LearningBayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th International Conference on Machine Learning, pages 681-688.

20. . L Wu, C Hsieh, J Sharpnack, Wu, L., Hsieh, C., and Sharpnack, J. (2018).

21. A listwise approach to collaborative ranking. Sql-Rank, Proceedings of the 35th International Conference on Machine Learning, ICML '18. the 35th International Conference on Machine Learning, ICML '18SQL-Rank: A listwise approach to collaborative ranking. In Proceedings of the 35th International Conference on Machine Learning, ICML '18, pages 5311-5320.

22. . Q Yang, Y Liu, T Chen, Y Tong, Yang, Q., Liu, Y., Chen, T., and Tong, Y. (2019).

23. Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology. 102Federated machine learning: Concept and applications. ACM Transactions on Intelligent Systems and Technology, 10(2):12:1-12:19.

24. Joint intelligence ranking by federated multiplicative update. C Zhang, Y Liu, L Wang, Y Liu, L Li, N Zheng, IEEE Intelligent Systems. 354Zhang, C., Liu, Y., Wang, L., Liu, Y., Li, L., and Zheng, N. (2020). Joint intelligence ranking by federated multiplicative update. IEEE Intelligent Systems, 35(4):15-24.
