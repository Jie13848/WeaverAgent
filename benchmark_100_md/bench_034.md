# Mixup-Transformer: Dynamic Data Augmentation for NLP Tasks

```meta
corpus_id: 263892219
```

## Authors

- Lichao Sun james.lichao.sun@gmail.com 
Lehigh University
- Congying Xia 
University of Illinois at Chicago
- Wenpeng Yin wyin@salesforce.com 
Salesforce Research
- Tingting Liang liangtt@hdu.edu.cn 
Hangzhou Dianzi University
- Philip S Yu psyu@uic.edu 
University of Illinois at Chicago
- Lifang He 
Lehigh University
- Lichao Sun james.lichao.sun@gmail.com 
Lehigh University
- Congying Xia 
University of Illinois at Chicago
- Wenpeng Yin wyin@salesforce.com 
Salesforce Research
- Tingting Liang liangtt@hdu.edu.cn 
Hangzhou Dianzi University
- Philip S Yu psyu@uic.edu 
University of Illinois at Chicago
- Lifang He 
Lehigh University

## Abstract

Mixup (Zhang et al., 2017)is a latest data augmentation technique that linearly interpolates input examples and the corresponding labels. It has shown strong effectiveness in image classification by interpolating images at the pixel level. Inspired by this line of research, in this paper, we explore: i) how to apply mixup to natural language processing tasks since text data can hardly be mixed in the raw format; ii) if mixup is still effective in transformer-based learning models, e.g., BERT. To achieve the goal, we incorporate mixup to transformer-based pre-trained architecture, named "mixup-transformer", for a wide range of NLP tasks while keeping the whole end-to-end training system. We evaluate the proposed framework by running extensive experiments on the GLUE benchmark. Furthermore, we also examine the performance of mixup-transformer in low-resource scenarios by reducing the training data with a certain ratio. Our studies show that mixup is a domain-independent data augmentation technique to pre-trained language models, resulting in significant performance improvement for transformer-based models.

## Introduction

## Introduction

Deep learning has shown outstanding performance in the field of natural language processing (NLP). Recently, transformer-based methods (Devlin et al., 2018;Yang et al., 2019) have achieved state-ofthe-art performance across a wide variety of NLP tasks 1 . However, these models highly rely on the availability of large amounts of annotated data, which is expensive and labor-intensive. To solve the data scarcity problem, data augmentation is commonly used in NLP tasks. For example, Wei and Zou (2019) investigated language transformations like insertion, deletion and swap. Several works (Malandrakis et al., 2019;Yoo et al., 2019;Xia et al., 2020b;Xia et al., 2020a) utilized variational autoencoders (VAEs) (Kingma and Welling, 2013) to generate more raw inputs. Nevertheless, these methods often rely on some extra knowledge to guarantee the quality of new inputs, and they have to be working in a pipeline. Zhang et al. (2017) proposed mixup, a domain-independent data augmentation technique that linearly interpolates image inputs on the pixel-based feature space. Guo et al. (2019) tried mixup in CNN (LeCun et al., 1998) and LSTM (Hochreiter and Schmidhuber, 1997) for text applications. Despite effectiveness, they conducted mixup only on the fixed word embedding level like Zhang et al. (2017) did in image classification. Two questions arise, therefore: (i) how to apply mixup to NLP tasks if text data cannot be mixed in the raw format? Apart from the embedding feature space, what other representation spaces can be constituted and used? ii) whether or not mixup can boost the state-of-the-art further in transformer-based learning models, such as BERT (Devlin et al., 2018).
Deep learning has shown outstanding performance in the field of natural language processing (NLP). Recently, transformer-based methods (Devlin et al., 2018;Yang et al., 2019) have achieved state-ofthe-art performance across a wide variety of NLP tasks 1 . However, these models highly rely on the availability of large amounts of annotated data, which is expensive and labor-intensive. To solve the data scarcity problem, data augmentation is commonly used in NLP tasks. For example, Wei and Zou (2019) investigated language transformations like insertion, deletion and swap. Several works (Malandrakis et al., 2019;Yoo et al., 2019;Xia et al., 2020b;Xia et al., 2020a) utilized variational autoencoders (VAEs) (Kingma and Welling, 2013) to generate more raw inputs. Nevertheless, these methods often rely on some extra knowledge to guarantee the quality of new inputs, and they have to be working in a pipeline. Zhang et al. (2017) proposed mixup, a domain-independent data augmentation technique that linearly interpolates image inputs on the pixel-based feature space. Guo et al. (2019) tried mixup in CNN (LeCun et al., 1998) and LSTM (Hochreiter and Schmidhuber, 1997) for text applications. Despite effectiveness, they conducted mixup only on the fixed word embedding level like Zhang et al. (2017) did in image classification. Two questions arise, therefore: (i) how to apply mixup to NLP tasks if text data cannot be mixed in the raw format? Apart from the embedding feature space, what other representation spaces can be constituted and used? ii) whether or not mixup can boost the state-of-the-art further in transformer-based learning models, such as BERT (Devlin et al., 2018).
To answer these questions, we stack a mixup layer over the final hidden layer of the pre-trained transformer-based model. The resulting system can be applied to a broad of NLP tasks; in particular, it is still end-to-end trainable. We evaluate our proposed mixup-transformer on the GLUE benchmark, which shows that mixup can consistently improve the performance of each task. Our contributions are summarized as follows:
To answer these questions, we stack a mixup layer over the final hidden layer of the pre-trained transformer-based model. The resulting system can be applied to a broad of NLP tasks; in particular, it is still end-to-end trainable. We evaluate our proposed mixup-transformer on the GLUE benchmark, which shows that mixup can consistently improve the performance of each task. Our contributions are summarized as follows:
• We propose the mixup-transformer that applies mixup into transformer-based pre-trained models. To our best knowledge, this is the first work that explores the effectiveness of mixup in Transformer.
• We propose the mixup-transformer that applies mixup into transformer-based pre-trained models. To our best knowledge, this is the first work that explores the effectiveness of mixup in Transformer.
• In experiments, we demonstrate that mixup-transformer can consistently promote the performance across a wide range of NLP benchmarks, and it is particularly helpful in low-resource scenarios where we reduce the training data from 10% to 90%.
• In experiments, we demonstrate that mixup-transformer can consistently promote the performance across a wide range of NLP benchmarks, and it is particularly helpful in low-resource scenarios where we reduce the training data from 10% to 90%.

## Mixup-Transformer

## Mixup-Transformer

In this section, we first introduce the mixup used in previous works. Then, we show how to incorporate the mixup into transformer-based methods and how to do the fine-turning on different text classification tasks. Last, we will discuss the difference between the previous works and our new approach.
In this section, we first introduce the mixup used in previous works. Then, we show how to incorporate the mixup into transformer-based methods and how to do the fine-turning on different text classification tasks. Last, we will discuss the difference between the previous works and our new approach.

## Mixup

## Mixup

Mixup is first proposed for image classification (Zhang et al., 2017), which incorporates the prior knowledge that linear interpolations of feature representations should lead to the same interpolations of the associated targets. In mixup, virtual training examples are constructed by two examples drawn at random from the training data:
Mixup is first proposed for image classification (Zhang et al., 2017), which incorporates the prior knowledge that linear interpolations of feature representations should lead to the same interpolations of the associated targets. In mixup, virtual training examples are constructed by two examples drawn at random from the training data:
where λ could be either fixed value in [0, 1] or λ ∼ Beta(α, α), for α ∈ (0, ∞). In previous works, mixup is a static data augmentation approach that improves and robusts the performance in image classification.
where λ could be either fixed value in [0, 1] or λ ∼ Beta(α, α), for α ∈ (0, ∞). In previous works, mixup is a static data augmentation approach that improves and robusts the performance in image classification.

## Mixup for Text Classification

## Mixup for Text Classification

Text classification is the most fundamental problem in the NLP field. Unlike image data, text input consists of discrete units (words) without an inherent ordering or algebraic operations -it could be one sentence, two sentences, a paragraph or a whole document.
Text classification is the most fundamental problem in the NLP field. Unlike image data, text input consists of discrete units (words) without an inherent ordering or algebraic operations -it could be one sentence, two sentences, a paragraph or a whole document.
The first step of text classification is to use the word embedding to convert each word of the text into a vector representation. In the traditional approaches, the word embedding method can be bag-of-words, or a fixed word to vector mapping dictionary built by CNN or LSTM. In our approach, instead of using the traditional encoding methods, we use transformer-based pre-trained language models to learn the representations for text data. For downstream tasks, we fine-tune transformer-based models with the mixup data augmentation method. Formally, mixup-transformer constructs virtual hidden representations dynamically durning the training process as follows:
The first step of text classification is to use the word embedding to convert each word of the text into a vector representation. In the traditional approaches, the word embedding method can be bag-of-words, or a fixed word to vector mapping dictionary built by CNN or LSTM. In our approach, instead of using the traditional encoding methods, we use transformer-based pre-trained language models to learn the representations for text data. For downstream tasks, we fine-tune transformer-based models with the mixup data augmentation method. Formally, mixup-transformer constructs virtual hidden representations dynamically durning the training process as follows:
where T (·) represents outputs of the transformer layers as shown in Figure 1. Note that, the mixup process is trained together with the fine-tuning process in an end-to-end fashion, and the hidden mixup representations are dynamic during the training process.
where T (·) represents outputs of the transformer layers as shown in Figure 1. Note that, the mixup process is trained together with the fine-tuning process in an end-to-end fashion, and the hidden mixup representations are dynamic during the training process.

## Discussion

## Discussion

In this section, we highlight two main differences between our approach and previous methods using mixup techniques (Zhang et al., 2017;Guo et al., 2019) for comparison.
In this section, we highlight two main differences between our approach and previous methods using mixup techniques (Zhang et al., 2017;Guo et al., 2019) for comparison.
• Dynamic mixup representation. For each input pair, x i and x j , the previous approaches produce a fixed mixup representation given a fixed λ. However, the mixup hidden representations in our approach are dynamic since they are trained together with the fine-tuning process.
• Dynamic mixup representation. For each input pair, x i and x j , the previous approaches produce a fixed mixup representation given a fixed λ. However, the mixup hidden representations in our approach are dynamic since they are trained together with the fine-tuning process.
• Dynamic mixup activation. Since a pre-trained network needs to be fine-tuned for a specific task, we can dynamically activate the mixup during the training. For example, if the training epoch is 3, we can choose to use mixup in any epoch or all epochs. In our experiments, we fine-tune the model without mixup in the first half of epochs for good representations and add mixup in the last half of the epochs.
• Dynamic mixup activation. Since a pre-trained network needs to be fine-tuned for a specific task, we can dynamically activate the mixup during the training. For example, if the training epoch is 3, we can choose to use mixup in any epoch or all epochs. In our experiments, we fine-tune the model without mixup in the first half of epochs for good representations and add mixup in the last half of the epochs.

## Experiments

## Experiments

To show the effectiveness of our proposed mixup-transformer, we conduct extensive experiments by adding the mixup strategy to transformer-based models on seven NLP tasks contained in the GLUE benchmark. Furthermore, we reduce the training data with different ratios (from 10% to 90%) to see how the mixup strategy works with insufficient training data. We report the performance on development sets for all the tasks because the test time is limited by the online GLUE benchmark. Baselines. Two baselines are conducted in the experiments, including BERT-base and BERT-large (Devlin et al., 2018). We evaluate the performance of our methods by adding the mixup strategy to these two baselines.
To show the effectiveness of our proposed mixup-transformer, we conduct extensive experiments by adding the mixup strategy to transformer-based models on seven NLP tasks contained in the GLUE benchmark. Furthermore, we reduce the training data with different ratios (from 10% to 90%) to see how the mixup strategy works with insufficient training data. We report the performance on development sets for all the tasks because the test time is limited by the online GLUE benchmark. Baselines. Two baselines are conducted in the experiments, including BERT-base and BERT-large (Devlin et al., 2018). We evaluate the performance of our methods by adding the mixup strategy to these two baselines.
Implementation details. When fine-tuning BERT with or without the mixup strategy for these NLP tasks, we fix the hyper-parameters as follows: the batch size is 8, the learning rate is 2e-5, the max sequence length is 128, and the number of the training epochs is 3. We test different values of λ, (from 0.1 to 0.9) on the default dataset (CoLA) and find mixup-transformer is insensitive to this hyper-parameter, so we set a fixed value of λ = 0.5. Experimental results for eight different NLP tasks are illustrated in Table 1. By adding the proposed mixup technique to BERT-base and BERT-large, the mixup-transformer improves the performance consistently on most of these tasks. The average improvement is around 1% for all the settings. The highest performance gain comes from the RTE task by adding the proposed mixup technique on BERT-base. In this experiment, the accuracy improves from 68.23% to 71.84%, which is an increase of 3.61%. The Matthew's correlation for CoLA increases from 59.71% to 62.39% (improved 2.68%) with mixup on BERT-large. Some experiments also get performance decrease with mixup. For example, adding mixup to BERT-base on STS-B decreases the Spearman correlation from 89.41% to 88.66%. Overall, most of the tasks improved (14 out of 16, while 2 got slightly worse) were found by applying mixup-transformer.
Implementation details. When fine-tuning BERT with or without the mixup strategy for these NLP tasks, we fix the hyper-parameters as follows: the batch size is 8, the learning rate is 2e-5, the max sequence length is 128, and the number of the training epochs is 3. We test different values of λ, (from 0.1 to 0.9) on the default dataset (CoLA) and find mixup-transformer is insensitive to this hyper-parameter, so we set a fixed value of λ = 0.5. Experimental results for eight different NLP tasks are illustrated in Table 1. By adding the proposed mixup technique to BERT-base and BERT-large, the mixup-transformer improves the performance consistently on most of these tasks. The average improvement is around 1% for all the settings. The highest performance gain comes from the RTE task by adding the proposed mixup technique on BERT-base. In this experiment, the accuracy improves from 68.23% to 71.84%, which is an increase of 3.61%. The Matthew's correlation for CoLA increases from 59.71% to 62.39% (improved 2.68%) with mixup on BERT-large. Some experiments also get performance decrease with mixup. For example, adding mixup to BERT-base on STS-B decreases the Spearman correlation from 89.41% to 88.66%. Overall, most of the tasks improved (14 out of 16, while 2 got slightly worse) were found by applying mixup-transformer.

## Results on full data

## Results on full data

## Results on limited data

## Results on limited data

As mixup is a technique for augmenting the feature space, it is interesting to see how it works when the training data is insufficient. Therefore, we reduce the training data with a certain ratio (from 10% to 90% with a step 10%) and test the effectiveness of mixup-transformer in low-resource scenarios. As shown in Table 2, BERT-large + mixup consistently outperforms BERT-large when we reduce the training data for MRPC, where the highest improvement (4.90%) is achieved when only using 40% of the training data.
As mixup is a technique for augmenting the feature space, it is interesting to see how it works when the training data is insufficient. Therefore, we reduce the training data with a certain ratio (from 10% to 90% with a step 10%) and test the effectiveness of mixup-transformer in low-resource scenarios. As shown in Table 2, BERT-large + mixup consistently outperforms BERT-large when we reduce the training data for MRPC, where the highest improvement (4.90%) is achieved when only using 40% of the training data.
Using the full training data (100%) gets an increase of 2.46%, which indicates that mixup-transformer works even better with reduced annotations. We also report the experiments of reducing training data on other tasks, including STS-B, RTE and CoLA. As shown in Figure 2, the mixup-transformer again consistently improves the performance for all the experiments. The performance gains with less training data (like 10% for STS-B, CoLA, and RTE) are higher than using full training data since data augmentation is more effective when the annotations are insufficient. Therefore, the mixup strategy is highly helpful in most low-resource scenarios.
Using the full training data (100%) gets an increase of 2.46%, which indicates that mixup-transformer works even better with reduced annotations. We also report the experiments of reducing training data on other tasks, including STS-B, RTE and CoLA. As shown in Figure 2, the mixup-transformer again consistently improves the performance for all the experiments. The performance gains with less training data (like 10% for STS-B, CoLA, and RTE) are higher than using full training data since data augmentation is more effective when the annotations are insufficient. Therefore, the mixup strategy is highly helpful in most low-resource scenarios.

## Conclusion

## Conclusion

In this paper, we propose the mixup-transformer that incorporates a data augmentation technique called mixup into transformer-based models for NLP tasks. Unlike using the static mixup in previous works, our approach can dynamically construct new inputs for text classification. Extensive experimental results show that mixup-transformer can be dynamically used with a pre-trained model to achieve better performance on GLUE benchmark.
In this paper, we propose the mixup-transformer that incorporates a data augmentation technique called mixup into transformer-based models for NLP tasks. Unlike using the static mixup in previous works, our approach can dynamically construct new inputs for text classification. Extensive experimental results show that mixup-transformer can be dynamically used with a pre-trained model to achieve better performance on GLUE benchmark.

## Figure 1 :

## Figure 1 :

## Figure 2 :

## Figure 2 :

## Table 2 :

## Table 2 :

## Figures (text descriptions)

### Figure 1

The overall framework of mixup-transformer. xi, xj are two separate sentences, fed to the same Transformer T . T (xi) is the representation of the input xi, generated by T .x andŷ are the interpolated representation and label, respectively.

Figure 1 :
1The overall framework of mixup-transformer. xi, xj are two separate sentences, fed to the same Transformer T . T (xi) is the representation of the input xi, generated by T .x andŷ are the interpolated representation and label, respectively.

### Figure 2

The overall framework of mixup-transformer. xi, xj are two separate sentences, fed to the same Transformer T . T (xi) is the representation of the input xi, generated by T .x andŷ are the interpolated representation and label, respectively.

Figure 1 :
1The overall framework of mixup-transformer. xi, xj are two separate sentences, fed to the same Transformer T . T (xi) is the representation of the input xi, generated by T .x andŷ are the interpolated representation and label, respectively.

### Figure 3

Datasets. The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks. Experiments are conducted on eight tasks in GLUE: CoLA (Warstadt et al., 2019), SST-2 (Socher et al., 2013), MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Z. Chen and Zhao, 2018), MNLI(Williams et al., 2017), QNLI(Rajpurkar et al., 2016), RTE(Bentivogli et al., 2009).

Datasets. The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks. Experiments are conducted on eight tasks in GLUE: CoLA (Warstadt et al., 2019), SST-2 (Socher et al., 2013), MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Z. Chen and Zhao, 2018), MNLI(Williams et al., 2017), QNLI(Rajpurkar et al., 2016), RTE(Bentivogli et al., 2009).

### Figure 4

Datasets. The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks. Experiments are conducted on eight tasks in GLUE: CoLA (Warstadt et al., 2019), SST-2 (Socher et al., 2013), MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Z. Chen and Zhao, 2018), MNLI(Williams et al., 2017), QNLI(Rajpurkar et al., 2016), RTE(Bentivogli et al., 2009).

Datasets. The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks. Experiments are conducted on eight tasks in GLUE: CoLA (Warstadt et al., 2019), SST-2 (Socher et al., 2013), MRPC (Dolan and Brockett, 2005), STS-B (Cer et al., 2017), QQP (Z. Chen and Zhao, 2018), MNLI(Williams et al., 2017), QNLI(Rajpurkar et al., 2016), RTE(Bentivogli et al., 2009).

### Figure 5

mixup-transformer with BERT-large runs with reduced training data for four tasks: STS-B, MRPC, RTE and CoLA.

Figure 2 :
2mixup-transformer with BERT-large runs with reduced training data for four tasks: STS-B, MRPC, RTE and CoLA.

### Figure 6

mixup-transformer with BERT-large runs with reduced training data for four tasks: STS-B, MRPC, RTE and CoLA.

Figure 2 :
2mixup-transformer with BERT-large runs with reduced training data for four tasks: STS-B, MRPC, RTE and CoLA.

### Figure 7

mixup-transformer with reduced training data on MRPC. Accuracy scores are used to evaluate the performance.

Table 2 :
2mixup-transformer with reduced training data on MRPC. Accuracy scores are used to evaluate the performance.

### Figure 8

mixup-transformer with reduced training data on MRPC. Accuracy scores are used to evaluate the performance.

Table 2 :
2mixup-transformer with reduced training data on MRPC. Accuracy scores are used to evaluate the performance.

## References

1. The fifth pascal recognizing textual entailment challenge. Luisa Bentivogli, TACPeter Clark, TACIdo Dagan, TACDanilo Giampiccolo, TACLuisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. 2009. The fifth pascal recognizing textual entailment challenge. In TAC.

2. Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, Lucia Specia, arXiv:1708.00055Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprintDaniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. 2017. Semeval-2017 task 1: Se- mantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055. (DOI: arXiv:1708.00055)

3. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, arXiv:1810.04805Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprintJacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirec- tional transformers for language understanding. arXiv preprint arXiv:1810.04805. (DOI: arXiv:1810.04805)

4. Automatically constructing a corpus of sentential paraphrases. B William, Chris Dolan, Brockett, Proceedings of the Third International Workshop on Paraphrasing (IWP2005). the Third International Workshop on Paraphrasing (IWP2005)William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).

5. Augmenting data with mixup for sentence classification: An empirical study. Hongyu Guo, Yongyi Mao, Richong Zhang, arXiv:1905.08941arXiv preprintHongyu Guo, Yongyi Mao, and Richong Zhang. 2019. Augmenting data with mixup for sentence classification: An empirical study. arXiv preprint arXiv:1905.08941. (DOI: arXiv:1905.08941)

6. Long short-term memory. Sepp Hochreiter, Jürgen Schmidhuber, Neural computation. 98Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735-1780.

7. P Diederik, Max Kingma, Welling, arXiv:1312.6114Auto-encoding variational bayes. arXiv preprintDiederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114. (DOI: arXiv:1312.6114)

8. Gradient-based learning applied to document recognition. Yann Lecun, Léon Bottou, Yoshua Bengio, Patrick Haffner, Proceedings of the IEEE. 8611Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. Gradient-based learning applied to docu- ment recognition. Proceedings of the IEEE, 86(11):2278-2324.

9. Controlled text generation for data augmentation in intelligent artificial agents. Nikolaos Malandrakis, Minmin Shen, Anuj Goyal, Shuyang Gao, Abhishek Sethi, Angeliki Metallinou, arXiv:1910.03487arXiv preprintNikolaos Malandrakis, Minmin Shen, Anuj Goyal, Shuyang Gao, Abhishek Sethi, and Angeliki Metallinou. 2019. Controlled text generation for data augmentation in intelligent artificial agents. arXiv preprint arXiv:1910.03487. (DOI: arXiv:1910.03487)

10. Squad: 100,000+ questions for machine comprehension of text. Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang, arXiv:1606.05250arXiv preprintPranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250. (DOI: arXiv:1606.05250)

11. Recursive deep models for semantic compositionality over a sentiment treebank. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, D Christopher, Manning, Y Andrew, Christopher Ng, Potts, Proceedings of the 2013 conference on empirical methods in natural language processing. the 2013 conference on empirical methods in natural language processingRichard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631-1642.

12. Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R Bowman, arXiv:1804.07461Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprintAlex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461. (DOI: arXiv:1804.07461)

13. Alex Warstadt, Amanpreet Singh, Samuel R Bowman, Neural network acceptability judgments. Transactions of the Association for Computational Linguistics. 7Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2019. Neural network acceptability judgments. Trans- actions of the Association for Computational Linguistics, 7:625-641.

14. W Jason, Kai Wei, Zou, arXiv:1901.11196Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprintJason W Wei and Kai Zou. 2019. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196. (DOI: arXiv:1901.11196)

15. A broad-coverage challenge corpus for sentence understanding through inference. Adina Williams, Nikita Nangia, Samuel R Bowman, arXiv:1704.05426arXiv preprintAdina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426. (DOI: arXiv:1704.05426)

16. Composed variational natural language generation for few-shot intents. Congying Xia, Caiming Xiong, Philip Yu, Richard Socher, arXiv:2009.10056arXiv preprintCongying Xia, Caiming Xiong, Philip Yu, and Richard Socher. 2020a. Composed variational natural language generation for few-shot intents. arXiv preprint arXiv:2009.10056. (DOI: arXiv:2009.10056)

17. Cg-bert: Conditional text generation with bert for generalized few-shot intent detection. Congying Xia, Chenwei Zhang, Hoang Nguyen, Jiawei Zhang, Philip Yu, arXiv:2004.01881arXiv preprintCongying Xia, Chenwei Zhang, Hoang Nguyen, Jiawei Zhang, and Philip Yu. 2020b. Cg-bert: Conditional text generation with bert for generalized few-shot intent detection. arXiv preprint arXiv:2004.01881. (DOI: arXiv:2004.01881)

18. Xlnet: Generalized autoregressive pretraining for language understanding. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, R Russ, Quoc V Salakhutdinov, Le, Advances in neural information processing systems. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information process- ing systems, pages 5753-5763.

19. Data augmentation for spoken language understanding via joint variational generation. Youhyun Kang Min Yoo, Sang-Goo Shin, Lee, Proceedings of the AAAI Conference on Artificial Intelligence. the AAAI Conference on Artificial Intelligence33Kang Min Yoo, Youhyun Shin, and Sang-goo Lee. 2019. Data augmentation for spoken language understanding via joint variational generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 7402-7409.

20. . X Zhang, Z Chen, H Zhang, L Zhao, Quora question pairsX. Zhang Z. Chen, H. Zhang and L. Zhao. 2018. Quora question pairs.

21. Hongyi Zhang, Moustapha Cisse, David Yann N Dauphin, Lopez-Paz, arXiv:1710.09412mixup: Beyond empirical risk minimization. arXiv preprintHongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2017. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412. (DOI: arXiv:1710.09412)

22. The fifth pascal recognizing textual entailment challenge. Luisa Bentivogli, TACPeter Clark, TACIdo Dagan, TACDanilo Giampiccolo, TACLuisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. 2009. The fifth pascal recognizing textual entailment challenge. In TAC.

23. Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, Lucia Specia, arXiv:1708.00055Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprintDaniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. 2017. Semeval-2017 task 1: Se- mantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055. (DOI: arXiv:1708.00055)

24. Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova, arXiv:1810.04805Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprintJacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. Bert: Pre-training of deep bidirec- tional transformers for language understanding. arXiv preprint arXiv:1810.04805. (DOI: arXiv:1810.04805)

25. Automatically constructing a corpus of sentential paraphrases. B William, Chris Dolan, Brockett, Proceedings of the Third International Workshop on Paraphrasing (IWP2005). the Third International Workshop on Paraphrasing (IWP2005)William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).

26. Augmenting data with mixup for sentence classification: An empirical study. Hongyu Guo, Yongyi Mao, Richong Zhang, arXiv:1905.08941arXiv preprintHongyu Guo, Yongyi Mao, and Richong Zhang. 2019. Augmenting data with mixup for sentence classification: An empirical study. arXiv preprint arXiv:1905.08941. (DOI: arXiv:1905.08941)

27. Long short-term memory. Sepp Hochreiter, Jürgen Schmidhuber, Neural computation. 98Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. Neural computation, 9(8):1735-1780.

28. P Diederik, Max Kingma, Welling, arXiv:1312.6114Auto-encoding variational bayes. arXiv preprintDiederik P Kingma and Max Welling. 2013. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114. (DOI: arXiv:1312.6114)

29. Gradient-based learning applied to document recognition. Yann Lecun, Léon Bottou, Yoshua Bengio, Patrick Haffner, Proceedings of the IEEE. 8611Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. 1998. Gradient-based learning applied to docu- ment recognition. Proceedings of the IEEE, 86(11):2278-2324.

30. Controlled text generation for data augmentation in intelligent artificial agents. Nikolaos Malandrakis, Minmin Shen, Anuj Goyal, Shuyang Gao, Abhishek Sethi, Angeliki Metallinou, arXiv:1910.03487arXiv preprintNikolaos Malandrakis, Minmin Shen, Anuj Goyal, Shuyang Gao, Abhishek Sethi, and Angeliki Metallinou. 2019. Controlled text generation for data augmentation in intelligent artificial agents. arXiv preprint arXiv:1910.03487. (DOI: arXiv:1910.03487)

31. Squad: 100,000+ questions for machine comprehension of text. Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, Percy Liang, arXiv:1606.05250arXiv preprintPranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250. (DOI: arXiv:1606.05250)

32. Recursive deep models for semantic compositionality over a sentiment treebank. Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, D Christopher, Manning, Y Andrew, Christopher Ng, Potts, Proceedings of the 2013 conference on empirical methods in natural language processing. the 2013 conference on empirical methods in natural language processingRichard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631-1642.

33. Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, Samuel R Bowman, arXiv:1804.07461Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprintAlex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461. (DOI: arXiv:1804.07461)

34. Alex Warstadt, Amanpreet Singh, Samuel R Bowman, Neural network acceptability judgments. Transactions of the Association for Computational Linguistics. 7Alex Warstadt, Amanpreet Singh, and Samuel R Bowman. 2019. Neural network acceptability judgments. Trans- actions of the Association for Computational Linguistics, 7:625-641.

35. W Jason, Kai Wei, Zou, arXiv:1901.11196Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprintJason W Wei and Kai Zou. 2019. Eda: Easy data augmentation techniques for boosting performance on text classification tasks. arXiv preprint arXiv:1901.11196. (DOI: arXiv:1901.11196)

36. A broad-coverage challenge corpus for sentence understanding through inference. Adina Williams, Nikita Nangia, Samuel R Bowman, arXiv:1704.05426arXiv preprintAdina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426. (DOI: arXiv:1704.05426)

37. Composed variational natural language generation for few-shot intents. Congying Xia, Caiming Xiong, Philip Yu, Richard Socher, arXiv:2009.10056arXiv preprintCongying Xia, Caiming Xiong, Philip Yu, and Richard Socher. 2020a. Composed variational natural language generation for few-shot intents. arXiv preprint arXiv:2009.10056. (DOI: arXiv:2009.10056)

38. Cg-bert: Conditional text generation with bert for generalized few-shot intent detection. Congying Xia, Chenwei Zhang, Hoang Nguyen, Jiawei Zhang, Philip Yu, arXiv:2004.01881arXiv preprintCongying Xia, Chenwei Zhang, Hoang Nguyen, Jiawei Zhang, and Philip Yu. 2020b. Cg-bert: Conditional text generation with bert for generalized few-shot intent detection. arXiv preprint arXiv:2004.01881. (DOI: arXiv:2004.01881)

39. Xlnet: Generalized autoregressive pretraining for language understanding. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, R Russ, Quoc V Salakhutdinov, Le, Advances in neural information processing systems. Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Russ R Salakhutdinov, and Quoc V Le. 2019. Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information process- ing systems, pages 5753-5763.

40. Data augmentation for spoken language understanding via joint variational generation. Youhyun Kang Min Yoo, Sang-Goo Shin, Lee, Proceedings of the AAAI Conference on Artificial Intelligence. the AAAI Conference on Artificial Intelligence33Kang Min Yoo, Youhyun Shin, and Sang-goo Lee. 2019. Data augmentation for spoken language understanding via joint variational generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pages 7402-7409.

41. . X Zhang, Z Chen, H Zhang, L Zhao, Quora question pairsX. Zhang Z. Chen, H. Zhang and L. Zhao. 2018. Quora question pairs.

42. Hongyi Zhang, Moustapha Cisse, David Yann N Dauphin, Lopez-Paz, arXiv:1710.09412mixup: Beyond empirical risk minimization. arXiv preprintHongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. 2017. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412. (DOI: arXiv:1710.09412)
