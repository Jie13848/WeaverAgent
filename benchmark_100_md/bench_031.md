# Learning to Attack Real-World Models for Person Re-identification via Virtual-Guided Meta-Learning

```meta
corpus_id: 235306549
```

## Authors

- Fengxiang Yang 
Artificial Intelligence Department
Xiamen University
China
- Zhun Zhong 
Department of Information Engineering and Computer Science
University of Trento
Italy
- Hong Liu 
National Institute of Informatics
Japan
- Zheng Wang 
The University of Tokyo
Japan
- Zhiming Luo zhiming.luo@xmu.edu 
Post Doctoral Mobile Station of Information and Communication Engineering
Xiamen University
China
- Shaozi Li 
Artificial Intelligence Department
Xiamen University
China
- Nicu Sebe 
Department of Information Engineering and Computer Science
University of Trento
Italy

Huawei Research
Ireland
- Shin ' Ichi Satoh 
National Institute of Informatics
Japan

The University of Tokyo
Japan

## Abstract

Recent advances in person re-identification (re-ID) have led to impressive retrieval accuracy.However, existing re-ID models are challenged by the adversarial examples crafted by adding quasi-imperceptible perturbations.Moreover, re-ID systems face the domain shift issue that training and testing domains are not consistent.In this study, we argue that learning powerful attackers with high universality that works well on unseen domains is an important step in promoting the robustness of re-ID systems.Therefore, we introduce a novel universal attack algorithm called "MetaAttack" for person re-ID.MetaAttack can mislead re-ID models on unseen domains by a universal adversarial perturbation.Specifically, to capture common patterns across different domains, we propose a meta-learning scheme to seek the universal perturbation via the gradient interaction between meta-train and meta-test formed by two datasets.We also take advantage of a virtual dataset (PersonX), instead of real ones, to conduct meta-test.This scheme not only enables us to learn with more comprehensive variation factors but also mitigates the negative effects caused by biased factors of real datasets.Experiments on three large-scale re-ID datasets demonstrate the effectiveness of our method in attacking re-ID models on unseen domains.Our final visualization results reveal some new properties of existing re-ID systems, which can guide us in designing a more robust re-ID model.Code and supplemental material are available at https://github.com/FlyingRoastDuck/MetaAttackAAAI21.

## Introduction

Person re-identification (re-ID) (Sun et al. 2018;Wang et al. 2018) aims to match pedestrians across non-overlapping cameras.Recent advances in person re-ID have witnessed great progress with the developments of deep models (Ye et al. 2020;Wang et al. 2020b).However, the robustness of deep re-ID models is challenged by the adversarial examples (Szegedy et al. 2014;Wang et al. 2020a).By disturbing images with quasi-imperceptible noises, re-ID models will suffer from catastrophic performance degradation.This makes the design of robust re-ID systems that are insensi-Figure 1: Schematic illustration of attacking in re-ID.Adversarial perturbation sets of real source, real target, and virtual (PersonX) datasets are visualized in different colors.Each perturbation set crushes the re-ID model on its corresponding dataset.The common region represents perturbations that can attack models on all datasets.This common region is hard to reach when directly training with only one dataset, e.g., optimizing with real source (δ init →δ real ) or PersonX (δ init →δ virtual ).Our MetaAttack leverages the interacted gradients from real source and PersonX to guide the initialized perturbation to the common region (δ init →δ meta ).tive to adversarial examples become an urgent issue to be resolved."Our strength grows out of our weakness."
-Ralph Waldo Emerson Inspired by this quote, in this work, we argue that learning powerful attackers helps verifying and improving the robustness of re-ID models, especially ones with high universality.Then, our goal is designing such an attacker, which will help reveal and understand the weaknesses of re-ID systems.
Most current studies in adversarial attack mainly focus on image classification (Moosavi-Dezfooli, Fawzi, and Frossard 2016;Moosavi-Dezfooli et al. 2017;Goodfellow, Shlens, and Szegedy 2015), while few (Tolias, Radenovic, and Chum 2019;Li et al. 2019;Wang et al. 2020a) have touched upon the attacking scheme of image retrieval, especially person re-ID.Different from image classification, 1) person re-ID is an open-set (Panareda Busto and Gall 2017) problem in which identities in the training and testing sets are non-overlapped, and 2) person re-ID often encounters a large domain shift issue that training and testing sets are from different domains (Zhong et al. 2018).Hence, it is important to learn an adversarial attacker that is appropriate for different person identities and can generalize to different unseen domains.Recently, MisRank (Wang et al. 2020a) focuses on attacking unseen domains by simulating the impact of image scale with a multi-scale feature extractor.Despite its effectiveness, MisRank has two shortcomings: 1) it requires to generate a unique noise for each query image, limiting the efficiency and flexibility; 2) it ignores various factors (e.g., illumination and viewpoint) that significantly influence the testing results, which should also be considered in generalization.In this paper, we aim to design an attacker that is 1) efficient and flexible, and 2) robust to more variations that may exist in unseen domains.
For the first aspect, we propose to adopt Universal Adversarial Perturbation (UAP) (Moosavi-Dezfooli et al. 2017) for person re-ID.The goal of UAP is to mislead models with a single universal perturbation, which can speed up the attacking process.In addition, the learned UAP can reflect the distribution bias of data (Moosavi-Dezfooli et al. 2017), which benefits the interpretation of the model and the proposal of a defense scheme.However, since the adversarial perturbation is, in fact, a kind of feature (Ilyas et al. 2019), it may excessively focus on the biased factors of training data and as such may fail to achieve good performance on unseen domains.In this study, we assume that there exists a universal perturbation that captures common factors across domains and can attack most domains.Taking Fig. 1 as an example, we consider the perturbation sets of different domains have intersections with each other.The "common region" represents the perturbations that can attack most domains and we aim to learn a perturbation belonging to it.
To achieve this, a straightforward way is to train with a much larger and comprehensive dataset.However, in practice, due to the difficulty of labeling, the size of existing datasets are limited.Although we can directly train with the combination of existing datasets, the domain shifts between different datasets will hamper capturing the shared knowledge among datasets.In addition, due to the data privacy, we can actually only access few datasets.To solve this problem, we propose to utilize the meta-learning (Finn, Abbeel, and Levine 2017) to simulate the cross-domain process using two datasets during training.The meta-learning separates the training data into meta-train and meta-test, which enables us to learn the basic knowledge from meta-train as well as generalize the variation factors from meta-test.Intuitively, if a meta-test can cover as many common factors as possible, then the learned perturbations would be potentially located at the "common region" and thus can well attack more unseen domains.However, the existing real-world datasets are seriously limited by the biased environmental factors (Sun and Zheng 2019), hindering the strength of the meta-learning.Therefore, for the second aspect, we consider to take a virtual (synthetic) dataset, PersonX (Sun and Zheng 2019), into the meta-training process.PersonX is designed to simulate important variation factors of the re-ID system, such as pose, viewpoint, illumination and background.Moreover, more factors can be involved by simply controlling and generating data with a unity engine.It is easier for virtual dataset to consider more environmental factors.Based on these, PersonX is more appropriate to be the meta-test for capturing common patterns across domains during meta-learning.
This work proposes a novel universal adversarial perturbation method, named "MetaAttack", for person re-ID through virtual-guided meta-learning.In our method, we use two datasets during training: a real dataset regarded as the source domain and a virtual dataset (PersonX) regarded as the extra association domain.MetaAttack is designed to learn a universal perturbation that disturbs queries and causes significant performance drop on different unseen (target) domains without any online modification.Specifically, we take the real dataset as meta-train and PersonX as meta-test.This enables us to simulate the cross-domain constraint and to learn the universal perturbation with a metalearning.During optimization, the gradient from meta-train and meta-gradient from meta-test are aggregated to obtain the final gradient, which is used to update the perturbation and can improve the universality of the learned perturbation.To sum up, our contributions mainly lie in three aspects:
• We propose a meta-learning scheme to learn universal perturbation for person re-ID.With our method, the crossdomain constraint is explicitly injected into the optimization, improving universality of the learned perturbation.
• We adopt a virtual dataset as meta-test during metaoptimization.The diverse, balanced virtual data enable us to capture more common patterns across domains.
• Extensive experiments on three large-scale benchmarks demonstrate the effectiveness of the proposed MetaAttack.Our method can outperform state-of-the-art approaches when attacking unseen domains, even using a smaller perturbation budget .
2 Related Work  methods focus on misleading classification models.Different from them, this paper concentrates on fooling person re-ID systems with a universal perturbation.

## Attack Person Re-ID System

There are few existing works that contribute to the attack of image retrieval problem (Li et al. 2019;Wang et al. 2020a), especially person re-ID.Li et al. (2019) design an attack scheme for image retrieval by corrupting label-wise, pairwise and list-wise relationships in the training set.Zheng et al. (2018) and Bai et al. (2019) study the effectiveness of different attack methods for person re-ID.The above methods do not explicitly consider the context of universal attack, which aims to attack unseen domains during testing and is the goal of this work.Our work is most related to the method in (Wang et al. 2020a) that designs a generator to produce perturbation and verify the universality of the generated perturbation.Different from their work, our method does not require any generators and adopts a virtual-guided meta-learning scheme to learn a UAP.

## Meta Learning

Meta-learning is designed to learn new tasks with limited training samples and improve the generalization to different tasks (Li et al. 2018;Guo et al. 2020).Existing metalearning methods can be mainly divided into three classes: metric-based (Snell, Swersky, and Zemel 2017;Sung et al. 2018), model-based (Santoro et al. 2016) and optimizingbased methods (Finn, Abbeel, and Levine 2017;Nichol and Schulman 2018).Our algorithm is constructed based on MAML (Finn, Abbeel, and Levine 2017), which is an optimizing-based method.MAML attempts to obtain a good initialized weight that can fast adapt to new tasks by simulating the learning process of new tasks with meta-test.Different from MAML that focuses on few-shot learning problem, this work aims to learn a universal perturbation that can be used for misleading re-ID models on unseen domains.

## Overall Framework

In Fig. 2, we show the overall framework of the proposed MetaAttack.In the training stage, we propose to optimize δ by meta-learning with a source dataset and an extra association dataset.The source data is a real dataset (e.g., Duke), which is adopted as the meta-train for basic optimization.
The extra association dataset is a virtual dataset (PersonX) that is utilized as meta-test to mimic possible real-world scenarios and improve the universality of δ.Our method tries to learn a δ locating at the "common region" that can successfully attack different domains.In the attack stage, the obtained δ fools re-ID models, resulting in incorrect ranking lists.Next, we will introduce our method in detail.

## Basic Losses for Attacking Re-ID Models

In this work, we aim to cheat re-ID models with a single universal perturbation.We use pair-wise and label-wise relations among training samples for the perturbation learning.
Misleading Pair-wise Relations.We follow (Wang et al. 2020a), which applies triplet loss to pull dissimilar pairs close and push similar pairs away.Different from (Wang et al. 2020a), we do not use the labels of training data to estimate the pair-wise relations between samples.Instead, we apply the centroids generated by clustering, which can better reveal the sample similarities of re-ID models (Li et al. 2019;Radenović, Tolias, and Chum 2018).
Optimizing perturbations with cluster centroids can be regarded as directly corrupting the feature space of the re-ID model.Therefore, we first quantize the training data into k centroids by the k-means algorithm with the features generated by re-ID models.Then, based on the obtained cluster centroids, we use the triplet loss to mislead pair-wise relations between samples, which is formulated as:
where [•] + is the max(•, 0) function.f ∈ R d×1 is the disturbed feature of the disturbed image I (f = F(I )), and d is the dimension of feature.c p and c n are closest and furthest cluster centroids of original image feature in the training data, respectively.m is the hyper-parameter that controls the margin of positive and negative pairs.Misleading Label-wise Relations.We also use the loss function proposed in (Li et al. 2019) to mislead label-wise relations among training samples.In this function, a misclassification loss is used to enforce a sample away from its nearest centroid while pull it close to its second-nearest centroid.The mis-classification loss is formulated as:
where c 1 and c 2 represent the nearest and second-nearest centroids of original image feature, respectively.

## Virtual-Guided Meta-Learning

To improve the universality of perturbation δ, an intuitive way is to train with a larger dataset.This is, however, not applicable in the real world because of the difficulty of labeling re-ID data and data privacy.We propose to generalize perturbation with meta-learning.In our method, the training data is formed by two datasets that are regarded as meta-train M tr and meta-test M te , respectively.As shown in Fig. 3, the final gradient for meta-optimization is obtained by combining gradients from both M tr and M te , which calibrates the universal perturbation δ to the direction that can perform well on both parts.Intuitively, if a meta-test set includes as many common factors as possible, the learned perturbation will more easily be located at the "common region," as shown in Fig. 1, and thus the perturbation has a better universality.To achieve this, we require a dataset that has more common and balanced factors to form M te .PersonX is a virtual synthetic dataset that contains several important and comprehensive variation factors of re-ID system.Hence PersonX is more appropriate for meta-learning.We present a virtual-guided meta-learning algorithm for attacking re-ID models.The algorithm is summarized in Alg. 1, which contains three steps.
Step 1: Meta-train.We utilize the source dataset S as the meta-train set and learn the perturbation δ with the loss functions introduced in Sec.3.2.For the given meta-train batch with N b samples, we perturb and extract their features with the re-ID model trained on the source data.The loss function for meta-train is formulated as:
: An illustration of our meta-learning process.Given current perturbation δ, we update it with gradient computed on meta-train data M tr and obtain temporary δ .Then, we compute meta-gradient on meta-test data M te with δ .Note that the meta-gradient is obtained from δ rather than δ .The final gradient is the combination of gradient and metagradient, which is used for updating δ.
where F mtr represents N b features of current disturbed meta-train batch, f i is the i-th feature, and λ is the balancing parameter.We follow (Dong et al. 2018;Li et al. 2019) to obtain a updated temporary δ through stochastic gradient descent (SGD) with momentum, which is formulated as:
where g is the momentum for updating δ and g is the momentum.µ is the weight of momentum and α is the learning rate.clip(•) is the function that ensures the constraint ||δ|| ∞ ≤ .δ is the updated temporary perturbation, which will be used in the following meta-test step.
Step 2: Meta-test.We use PersonX to be the meta-test M te and compute meta-test loss with temporary δ obtained in
Step 1, which is defined as:
where F mte indicates N b features of current disturbed metatest batch, f i is the i-th feature extracted by re-ID model.The meta-test is used to mimic the perturbation attack process on unseen target domains.We use the meta-test loss to calculate the meta-gradient on the original δ, which can be considered as a regularization term to guide the original δ to the "common region" in the final step.
Step 3: Meta-update.The final step is based on the losses in the aforementioned two steps.Specifically, our final metaloss function for learning δ is:
(6) The former item aims to learn basic knowledge with metatrain, while the latter item aims to capture common factors across domains that are helpful in improving universality.

## Universal Attack

The optimized δ is utilized to proceed universal attack, as shown in the right part of Fig. 2. We mainly aim to attack Algorithm 1 Procedure of MetaAttack.Inputs: Meta-train M tr (source dataset S), meta-test M te (association dataset), number of centroids k, batch size N b , re-ID model trained on source domain F S , maximum iterations max iter, learning rate α.Outputs: Universal perturbation δ.Compute final loss and update δ through SGD with momentum (Eq.6); 14:
until M tr and M te are enumerated; 15: end for 16: Return δ; re-ID models of unseen domains.This goal is achieved by directly adding δ to all queries and corrupting their corresponding retrieval ranking lists.

## Experiments

Datasets.We use three large-scale re-ID benchmarks to verify our algorithm, i.e.Market-1501 (Market) (Zheng et al. 2015), DukeMTMC-reID (Duke) (Ristani et al. 2016;Zheng, Zheng, and Yang 2017) and MSMT-17 (MSMT) (Wei et al. 2018).Market contains 32, 668 images of 1, 501 identities obtained from six cameras.Duke consists of 36, 411 labeled images of 1, 404 identities pictured by eight different cameras.MSMT has 126, 441 images from 4, 101 pedestrians captured by fifteen cameras.For each dataset, nearly half of the identities are used for training.We only use PersonX-456 as meta-test, which removes all samples without backgrounds in PersonX (Sun and Zheng 2019) and contains 39, 852 images from 410 identities.Evaluation Protocol.To show the universality of different attack methods, we learn δ on a source dataset and then adopt δ to corrupt queries of other (target) datasets.In this paper, only the real datasets will be used as source and target datasets.The virtual dataset (PersonX) is an extra association dataset for our MetaAttack.The widely used mAP and rank-1 accuracy are used for evaluation.Lower mAP and rank-1 accuracies indicate better attack performance.Experimental Settings.We test our method on both global-based and part-based models.For the first, we use IDE (Zheng, Yang, and Hauptmann 2016) to train the re-ID model and extract pooling-5 feature to compute Eq. 6 for meta-optimization.For the second, we use PCB (Sun et al. 2018) to train the re-ID model.Specifically, PCB considers pedestrians as six parts and extract 256-dim feature for each part. 1 We use the ResNet-50 (He et al. 2016) as the backbone for both models.
All hyper-parameters in our experiments are set as follow: the number of centroids k = 512, the batch size N b = 50, the iteration number max iter = 20, margin m = 0.5, and the learning rate α = /10.We use SGD with momentum (Dong et al. 2018;Li et al. 2019) to update δ, and the weight of momentum µ = 1.The balancing factor λ is set to 10.We perform L ∞ -bounded attacks with = 8 unless otherwise noted.is the upper bound for each pixel of the generated δ, i.e., ||δ|| ∞ ≤ .

## Comparison with State-of-the-Art

We first compare our method with two state-of-the-art algorithms: MisRank2 (Wang et al. 2020a) and UAP-Retrieval 2 (Li et al. 2019).In most experiments, we set the = 8 to obtain quasi-imperceptible perturbation.We also report results when = 16 for fair comparison with Mis-Rank (Wang et al. 2020a).In addition, since our method uses PersonX as the extra association dataset, we report the results of training MisRank with both source data and PersonX ("MisRank+PersonX").In Tab. 1, the first two columns of results (Duke → Market and Duke → MSMT) use Duke as the source domain and the other two datasets (Market and MSMT) as target domains.Similar settings are used for the last two columns of results.
From Tab. 1, we have the following conclusions.
(1) Our method can achieve the best attack results with the same in all settings.This demonstrates the effectiveness of our method in attacking unseen domains and shows that our method is capable of attacking both global-and part-based models.
(2) The effect of MisRank largely relies on a larger .When = 8, MisRank fails to achieve competitive attacking results while our method obtains reasonable results that clearly outperform MisRank.Importantly, our method with = 8 can obtain better results than MisRank with = 16 in some settings.For example, in the setting of Duke → Market, our method with = 8 reduces the mAP to 4.9%.This is 5.4% lower than MisRank with = 16.(3) PersonX can not bring improvement for MisRank.When additionally training with PersonX, the attacking results of MisRank are even worse compared to the one trained with only source data.This suggests that PersonX may be not suitable for generator-based method and that leveraging the extra virtual dataset is not trivial in attack re-ID.
Table 1: Results for attacking re-ID systems.We use our method to attack different backbones (IDE (Zheng, Yang, and Hauptmann 2016) and part-based PCB (Sun et al. 2018)), then compare our method with state-of-the-arts (MisRank (Wang et al. 2020a) and UAP-Retrieval (Li et al. 2019))."Before Attack": re-ID accuracies of unseen target model on target set.

## Visualization

In this section, we visualize the obtained δ and some perturbed query images to give an intuitive presentation of the proposed MetaAttack algorithm.
Robust Queries in MetaAttack.Our MetaAttack can effectively corrupt re-ID accuracies with slight modifications to query images.However, it remains several robust queries that can defend our attack.To find out their common attributes, we show some examples in Fig. 4(a) and (b).All the experiments in this part use the IDE model.We observe two kinds of situations that may help defend our attack.The first kind of robust query is caused by occlusion and we visualize its ranking lists before and after attack in the first and second rows of Fig. 4(c).Since there are few occluded samples in the training data, the learned perturbation will only capture the overall distribution of nonoccluded images but be sensitive to occluded images.Therefore, during testing, we can try to add an occluding process before forwarding queries to the re-ID model, for defending the adversarial samples, e.g., adding erasing (Carmon et al. 2019;Zhong et al. 2020).Another kind of robust query is caused by the camera shift in the dataset.As shown in the third (before attack) and fourth (after attack) rows of Fig. 4(c), the pedestrian in the query image with a green T-shirt, changes to a blue T-shirt in its correctly matched nearest neighbor.The change of appearance is caused by camera shift (Zhong et al. 2018) and has been a long-standing problem in person re-ID.Both the source data and the PersonX do not contain such kind of cases, which causes our failure.In fact, each domain may contain its own specific camera shift that is very different to other domains.Therefore, we argue that adding domain-specific camera shift to query images may help defend our attack, which can be used as a reference for designing defense models of re-ID.Similar results and conclusions can also be found in Duke (Fig. 4(d)).Visualizations of δ and Perturbed Images.In Fig. 5, we visualize the obtained δ and some perturbed images.Our method is more efficient and flexible than MisRank, because our method only requires a single perturbation for all queries while MisRank needs to generate new perturbations for dif-Table 4: SSIM scores of generated adversarial examples between (Wang et al. 2020a) and our method.

## Further Experiments

Image Quality.SSIM (Wang et al. 2004) is a kind of metric to measure the similarity of two images and has been widely used to evaluate the quality of GAN-made (Goodfellow et al. 2014) virtual images.A larger SSIM score between synthetic and natural images indicates better quality and less distortion.We, therefore, utilize SSIM to evaluate the degree of distortion for adversarial examples.We report SSIM scores in Tab. 4. Compared with MisRank, our method produces higher SSIM scores, indicating that our method can generate higher quality adversarial images and achieve better attack performance.Sensitive Analysis.We change the value of from 8 to 16 and study the influence of perturbation budget.In Fig. 6, we plot the curve of mAP and rank-1 scores under two settings (Market→Duke and Duke→Market).The results show that a larger can easily damage re-ID accuracies.However, to make the obtained perturbation quasi-imperceptible, we suggest using a small to attack real-world re-ID models if a good attack results can be achieved.

## Conclusion

In this paper, we propose a novel universal attack algorithm for person re-ID, which is based on the virtual-guided meta-learning.Our method takes the source dataset to be meta-train and the synthetic PersonX dataset as meta-test.By combining the gradients from both meta-train and metatest sets during meta-optimization, the obtained perturbation can learn to generalize in unseen target domains and achieve satisfactory results.The proposed method performs well on three large-scale datasets with both IDE and PCB models.In our future work, we consider applying our observations and perspectives to design robust re-ID that can defend against adversarial samples.

## Figure 2 :

Figure 2: The framework of the proposed MetaAttack.During training, we use the source dataset S as meta-train M tr and PersonX as meta-test M te to simulate cross-domain attack.The aggregation of gradients computed by meta-train and meta-test is used to optimize the perturbation δ.During testing, δ can attack both source domain and unseen target domains.
3 MethodologyProblem Definition.We aim to seek a universal adversarial perturbation δ that can mislead ranking results of re-ID models in both source domain S and unseen target domains T .The attack operation is achieved by adding δ to a query image I.The perturbed query I (I = I + δ) is used to retrieve from the gallery and mislead victim re-ID model F.

## 1:

Use k-means clustering to obtain k centroids on M tr and M te , respectively; 2: Initialize δ with 0; 3: for i in max iter do batches m tr and m te with N b images from M tr and M te , respectively; 6: Disturb m tr and m te with δ to obtain m tr and m te ; 7: Extract features for disturbed m tr and m te with F S to obtain F mtr ∈ R d×N b and F mte ∈ R d×N b ; meta-train loss and obtain temporary δ with F mtr (Eq. 3 and Eq.4); 10: // Step 2: Meta-test 11: Compute meta-test loss with δ and F mte (Eq.5); 12: // Step 3: Meta-update 13:

## Figure 5 :

Figure 4: The visualization of some robust queries.

## Figure 6 :

Figure 6: Sensitive analysis of .

## Table 2 :

Ablation study on the proposed virtual-guided meta-learning algorithm.

## Table 3 :

Results on source domain.

## Figures (text descriptions)

### Figure 1

Figure 2: The framework of the proposed MetaAttack.During training, we use the source dataset S as meta-train M tr and PersonX as meta-test M te to simulate cross-domain attack.The aggregation of gradients computed by meta-train and meta-test is used to optimize the perturbation δ.During testing, δ can attack both source domain and unseen target domains.

Figure 2 :
2
Figure 2: The framework of the proposed MetaAttack.During training, we use the source dataset S as meta-train M tr and PersonX as meta-test M te to simulate cross-domain attack.The aggregation of gradients computed by meta-train and meta-test is used to optimize the perturbation δ.During testing, δ can attack both source domain and unseen target domains.

### Figure 2

3 MethodologyProblem Definition.We aim to seek a universal adversarial perturbation δ that can mislead ranking results of re-ID models in both source domain S and unseen target domains T .The attack operation is achieved by adding δ to a query image I.The perturbed query I (I = I + δ) is used to retrieve from the gallery and mislead victim re-ID model F.

3 MethodologyProblem Definition.We aim to seek a universal adversarial perturbation δ that can mislead ranking results of re-ID models in both source domain S and unseen target domains T .The attack operation is achieved by adding δ to a query image I.The perturbed query I (I = I + δ) is used to retrieve from the gallery and mislead victim re-ID model F.

### Figure 3

Use k-means clustering to obtain k centroids on M tr and M te , respectively; 2: Initialize δ with 0; 3: for i in max iter do batches m tr and m te with N b images from M tr and M te , respectively; 6: Disturb m tr and m te with δ to obtain m tr and m te ; 7: Extract features for disturbed m tr and m te with F S to obtain F mtr ∈ R d×N b and F mte ∈ R d×N b ; meta-train loss and obtain temporary δ with F mtr (Eq. 3 and Eq.4); 10: // Step 2: Meta-test 11: Compute meta-test loss with δ and F mte (Eq.5); 12: // Step 3: Meta-update 13:

1:

Use k-means clustering to obtain k centroids on M tr and M te , respectively; 2: Initialize δ with 0; 3: for i in max iter do batches m tr and m te with N b images from M tr and M te , respectively; 6: Disturb m tr and m te with δ to obtain m tr and m te ; 7: Extract features for disturbed m tr and m te with F S to obtain F mtr ∈ R d×N b and F mte ∈ R d×N b ; meta-train loss and obtain temporary δ with F mtr (Eq. 3 and Eq.4); 10: // Step 2: Meta-test 11: Compute meta-test loss with δ and F mte (Eq.5); 12: // Step 3: Meta-update 13:

### Figure 4

Figure 4: The visualization of some robust queries.

Figure 5 :
5
Figure 4: The visualization of some robust queries.

### Figure 5

Figure 6: Sensitive analysis of .

Figure 6 :
6
Figure 6: Sensitive analysis of .

### Figure 6

Ablation study on the proposed virtual-guided meta-learning algorithm.

Table 2 :
2
Ablation study on the proposed virtual-guided meta-learning algorithm.
No.Duke → MSMT Market → MSMT Extra Data mAP rank-1 mAP rank-1 Real PersonX Learning Meta1 5.614.35.814.9×××2 5.114.55.714.3××3 4.810.45.012.6×4 4.69.95.514.2××5 3.58.33.48.3×4.2 Ablation StudyTo show the effectiveness of the proposed method, we con-duct experiments by adding extra training data and meta-learning into the baseline. Results with IDE model are re-ported in Tab. 2. The first row (No.1) is the baseline thatonly uses the basic losses functions (Sec. 3.2) on the train-ing data. For the extra real data, we use Market when usingDuke as the source domain, vice versa.The effectiveness of meta-learning. To verify the signif-icance of the meta-learning strategy, we compare with thevariant that directly trained with the source data and the ex-tra data. From the comparison of No.2 vs No.3 and No.4 vsNo.5, we can observe that 1) directly combing the sourceand extra data brings limited improvement; and 2) trainingwith the meta-learning strategy can consistently improve theattack results and universality of learned perturbation.The benefit of virtual data in meta-learning. Anotherimportant component of our method is adopting a virtualdataset instead of a real one during meta-learning. The com-parison of No.3 vs No.5 shows that virtual-guided meta-learning outperforms the real-guided one. For example,when using Duke as the source domain, virtual-guided meta-learning (No.5) reduces the mAP to 3.5%, which is lowerthan real-guided one (No.3) by 1.3%. These results indicatethat using a dataset with less biased factors can improve theuniversality of learned perturbation.

### Figure 7

Results on source domain.

Table 3 :
3
Results on source domain.
BackboneMethodDuke mAP rank-1Market mAP rank-1Before Attack66.780.978.288.7IDEUAP-Retrieval4.29.93.64.5Ours3.66.43.13.4Before Attack68.084.176.791.3PCBUAP-Retrieval14.320.310.715.1Ours11.216.510.915.44.3 Performance on Source DomainIn Tab. 3, we report results on the testing set of source do-main and compare our MetaAttack with UAP-Retrieval forboth IDE and PCB models. Tab. 3 shows that our modelcan effectively corrupt the accuracies of the ranking list onthe source domain, and can achieve better results than UAP-Retrieval in most settings. Since UAP-Retrieval uses almostthe same basic loss functions to our MetaAttack, it can be re-garded as the reduction of MetaAttack, which does not usevirtual-guided meta-learning. Then, we can conclude thatour MetaAttack can also improve the university of pertur-bation in the source domain.

## References

1. Metric Attack and Defense for Person Re-identification. S Bai, Y Li, Y Zhou, Q Li, P H Torr, arXiv:1901.106502019arXiv preprint (DOI: arXiv:1901.10650)

2. Unlabeled data improves adversarial robustness. Y Carmon, A Raghunathan, L Schmidt, J C Duchi, P S Liang, NeurIPS. 2019

3. Boosting adversarial attacks with momentum. Y Dong, F Liao, T Pang, H Su, J Zhu, X Hu, J Li, CVPR. 2018

4. Sparse adversarial attack via perturbation factorization. Y Fan, B Wu, T Li, Y Zhang, M Li, Z Li, Y Yang, ECCV. 2020

5. Model-agnostic metalearning for fast adaptation of deep networks. C Finn, P Abbeel, S Levine, ICML. 2017

6. Generative adversarial nets. I Goodfellow, J Pouget-Abadie, M Mirza, B Xu, D Warde-Farley, S Ozair, A Courville, Y Bengio, 2014In NeurIPS

7. Explaining and harnessing adversarial examples. I J Goodfellow, J Shlens, C Szegedy, 2015In ICLR

8. Learning meta face recognition in unseen domains. J Guo, X Zhu, C Zhao, D Cao, Z Lei, S Z Li, CVPR. 2020

9. Deep residual learning for image recognition. K He, X Zhang, S Ren, J Sun, CVPR. 2016

10. Adversarial examples are not bugs, they are features. A Ilyas, S Santurkar, D Tsipras, L Engstrom, B Tran, A Madry, NeurIPS. 2019

11. Learning to generalize: Meta-learning for domain generalization. D Li, Y Yang, Y.-Z Song, T M Hospedales, J Li, R Ji, H Liu, X Hong, Y Gao, Q Tian, AAAI. Iccv, A Madry, A Makelov, L Schmidt, D Tsipras, A Vladu, 2018. 2019. 2018ICLR

12. Universal adversarial perturbations. S.-M Moosavi-Dezfooli, A Fawzi, O Fawzi, P Frossard, CVPR. 2017

13. Deepfool: a simple and accurate method to fool deep neural networks. S.-M Moosavi-Dezfooli, A Fawzi, P Frossard, CVPR. 2016

14. Reptile: a scalable metalearning algorithm. A Nichol, J Schulman, arXiv:1803.029992018arXiv preprint (DOI: arXiv:1803.02999)

15. Open set domain adaptation. P Panareda Busto, J Gall, ICCV. 2017

16. Generative adversarial perturbations. O Poursaeed, I Katsman, B Gao, S Belongie, CVPR. 2018

17. Fine-tuning CNN image retrieval with no human annotation. F Radenović, G Tolias, O Chum, IEEE TPAMI. 4172018

18. Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking. E Ristani, F Solera, R Zou, R Cucchiara, C Tomasi, ECCV. 2016

19. Meta-learning with memory-augmented neural networks. A Santoro, S Bartunov, M Botvinick, D Wierstra, T Lillicrap, ICML. 2016

20. Prototypical networks for few-shot learning. J Snell, K Swersky, R Zemel, 2017In NeurIPS

21. One pixel attack for fooling deep neural networks. J Su, D V Vargas, K Sakurai, IEEE TEC. 2352019

22. Dissecting person re-identification from the viewpoint of viewpoint. X Sun, L Zheng, CVPR. 2019

23. Beyond part models: Person retrieval with refined part pooling (and a strong convolutional baseline). Y Sun, L Zheng, Y Yang, Q Tian, S Wang, ECCV. 2018

24. Learning to compare: Relation network for few-shot learning. F Sung, Y Yang, L Zhang, T Xiang, P H Torr, T M Hospedales, CVPR. 2018

25. Intriguing properties of neural networks. C Szegedy, W Zaremba, I Sutskever, J Bruna, D Erhan, I Goodfellow, R Fergus, 2014In ICLR

26. Learning discriminative features with multiple granularities for person reidentification. G Tolias, F Radenovic, O Chum, G Wang, Y Yuan, X Chen, J Li, X Zhou, ACM MM. 2019. 2018ICCV

27. Transferable, Controllable, and Inconspicuous Adversarial Attacks on Person Re-identification With Deep Mis-Ranking. H Wang, G Wang, Y Li, D Zhang, L Lin, CVPR. 2020a

28. Image quality assessment: from error visibility to structural similarity. Z Wang, A C Bovik, H R Sheikh, E P Simoncelli, IEEE TIP. 1342004

29. Beyond intra-modality: A survey of heterogeneous person re-identification. Z Wang, Z Wang, Y Zheng, Y Wu, W Zeng, S Satoh, IJCAI. 2020b

30. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. L Wei, S Zhang, W Gao, Q Tian, M Ye, J Shen, G Lin, T Xiang, L Shao, S C H Hoi, arXiv:2001.04193Deep Learning for Person Re-identification: A Survey and Outlook. 2018. 2020arXiv preprintCVPR. (DOI: arXiv:2001.04193)

31. Scalable Person Re-identification: A Benchmark. L Zheng, L Shen, L Tian, S Wang, J Wang, Q Tian, L Yang, Y Hauptmann, A G , arXiv:1610.02984Person re-identification: Past, present and future. 2015. 2016arXiv preprintCVPR (DOI: arXiv:1610.02984)

32. Open set adversarial examples. Z Zheng, L Zheng, Z Hu, Y Yang, arXiv:1809.026812018arXiv preprint (DOI: arXiv:1809.02681)

33. Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in vitro. Z Zheng, L Zheng, Y Yang, ICCV. 2017

34. Random Erasing Data Augmentation. Z Zhong, L Zheng, G Kang, S Li, Y Yang, AAAI. 2020

35. Camera style adaptation for person re-identification. Z Zhong, L Zheng, Z Zheng, S Li, Y Yang, CVPR. 2018
