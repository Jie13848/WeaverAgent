# DCFEE: A Document-level Chinese Financial Event Extraction System based on Automatically Labeled Training Data

```meta
corpus_id: 51871198
```

## Authors

- Hang Yang hang.yang@nlpr.ia.ac.cn 
Institute of Automation
National Laboratory of Pattern Recognition
Chinese Academy of Sciences
100190BeijingChina
- Yubo Chen yubo.chen@nlpr.ia.ac.cn 
Institute of Automation
National Laboratory of Pattern Recognition
Chinese Academy of Sciences
100190BeijingChina
- Kang Liu kliu@nlpr.ia.ac.cn 
Institute of Automation
National Laboratory of Pattern Recognition
Chinese Academy of Sciences
100190BeijingChina

University of Chinese Academy of Sciences
100049BeijingChina
- Yang Xiao yang.xiao@nlpr.ia.ac.cn 
Institute of Automation
National Laboratory of Pattern Recognition
Chinese Academy of Sciences
100190BeijingChina
- Jun Zhao jzhao@nlpr.ia.ac.cn 
Institute of Automation
National Laboratory of Pattern Recognition
Chinese Academy of Sciences
100190BeijingChina

University of Chinese Academy of Sciences
100049BeijingChina

## Abstract

We present an event extraction framework

## Introduction

Event Extraction (EE), a challenging task in Nature Language Processing (NLP), aims at discovering event mentions 3 and extracting events which contain event triggers 4 and event arguments 5 from texts. For example, in the sentence E1 6 as shown in Figure 1, an EE system is expected to discover an Equity Freeze event mention (E1 itself) triggered by frozen and extract the corresponding five arguments with different roles: Nagafu Ruihua (Role=Shareholder Name), 520,000 shares (Role=Num of Frozen Stock), People's Court of Dalian city (Role=Frozen Institution), May 5,2017 (Role=Freezing Start Date) and 3 years (Role=Freezing End Date). Extracting event instances from texts plays a critical role in building NLP applications such as Information Extraction (IE), Question Answer (QA) and Summarization (Ahn, 2006). Recently, researchers have built some English EE systems, such as EventRegistry 7 and Stela 8 . However, in financial domain, there is no such effective EE system, especially in Chinese.
Financial events are able to help users obtain competitors' strategies, predict the stock market and make correct investment decisions. For example, the occurrence of an Equity Freeze event will have a bad effect on the company and the shareholders should make correct decisions quickly to avoid the losses. In business domain, official announcements released by companies represent the occurrence of major events, such as Equity Freeze events, Equity Trading events and so on. So it is valuable to discover event mention and extract events from the announcements. However, there are two challenges in Chinese financial EE.
Lack of data: most of the EE methods usually adopted supervised learning paradigm which relies on elaborate human-annotated data, but there is no labeled corpus for EE in the Chinese financial field.
Document-level EE: most of the current methods of EE are concentrated on the sentence-level  Figure 1: Example of an Equity Freeze event triggered by "frozen" and containing five arguments.
text ) (Nguyen et al., 2016. But an event is usually expressed with multiple sentences in a document. In the financial domain data set constructed in this paper, there are 91% of the cases that the event arguments are distributed in the different sentences. For example, as shown in Figure 1, E1 and E2 describe an Equity Freeze event together.
To solve the above two problems, we present a framework named DCFEE which can extract document-level events from announcements based on automatically labeled training data. We make use of Distance Supervision (DS) which has been validated to generate labeled data for EE  to automatically generate large-scaled annotated data. We use a sequence tagging model to automatically extract sentence-level events. And then, we propose a key-event detection model and an arguments-filling strategy to extract the whole event from the document.
In summary, the contributions of this article are as follows:
• We propose the DCFEE framework which can automatically generate large amounts of labeled data and extract document-level events from the financial announcements.
• We introduce an automatic data labeling method for event extraction and give a series of useful tips for constructing Chinese financial event dataset. We propose a documentlevel EE system mainly based on a neural sequence tagging model, a key-event detection model, and an arguments-completion strategy. The experimental results show the effectiveness of it.
• The DCFEE system has been successfully built as an online application which can quickly extract events from the financial an-nouncements 9 .  2 Methodology Figure 2 describes the architecture of our proposed DCFEE framework which primarily involves the following two components: (i) Data Generation, which makes use of DS to automatically label event mention from the whole document (document-level data) and annotate triggers and arguments from event mention (sentence-level data); (ii) EE system, which contains Sentence-level Event Extraction (SEE) supported by sentencelevel labeled data and Document-level Event Extraction (DEE) supported by document-level labeled data. In the next section, we briefly describe the generation of labeled data and architecture of the EE system. Figure 3 describes the process of labeled data generation based on the method of DS. In this section, we first introduce the data sources (structured data and unstructured data) that we use. And then we describe the method of automatically labeling data. Finally, we will introduce some tips that can be used to improve the quality of the labeled data. Data sources: two types of data resources are required to automatically generate data: a financial event knowledge database containing a lot of structured event data and unstructured text data containing event information. (i) The financial event knowledge database used in this paper is structured data which includes nine common financial event types and is stored in a table format. These structured data which contains key event arguments is summarized from the announcements by financial professionals. An Equity Pledge event is taken as an example, as shown on the left of Figure 3, in which key arguments include Shareholder Name (NAME), Pledge Institution (ORG), Number of Pledged Stock (NUM), Pledging Start Date (BEG) and Pledging End Date (END). (ii) The unstructured text data come from official announcements released by the companies which are stored in an unstructured form on the web. We obtain these textual data from Sohu securities net 10 .

## Data Generation

Method of data generation: annotation data consists of two parts: sentence-level data generated by labeling the event trigger and event arguments in the event mention; document-level data generated by labeling the event mention from the document-level announcement. Now the question is, how to find the event triggers. Event arguments and event mention that correspond to the structured event knowledge database are summarized from a mass of announcements. DS has proved its effectiveness in automatically labeling data for Relation Extraction  and Event Extraction . Inspired by D-S, we assume that one sentence contains the most event arguments and driven by a specific trigger is likely to be an event mention in an announcement. And arguments occurring in the event mention are 10 http://q.stock.sohu.com/cn/000001/gsgg.shtml likely to play the corresponding roles in the event. For each type of financial event, we construct a dictionary of event triggers such as frozen in Equity Freeze event and pledged in Equity Pledge event. So the trigger word can be automatically marked by querying the pre-defined dictionary from the announcements. through these pretreatments, structured data can be mapped to the event arguments within the announcements. Therefore, we can automatically identify the event mention and label the event trigger and the event arguments contained in it to generate the sentence-level data, as shown at the bottom of Figure 3. Then, the event mention is automatically marked as a positive example and the rest of the sentences in the announcement are marked as negative examples to constitute the document-level data, as shown on the right of Figure 3. The document-level data and the sentence-level data together form the training data required for the EE system.
Tips: in reality, there are some challenges in data labeling: the correspondence of financial announcements and event knowledge base; the ambiguity and abbreviation of event arguments. There are some tips we used to solve these problems, examples are shown in Figure 3. (i) Decrease the search space: the search space of candidate announcements can be reduced through retrieving key event arguments such as the publish date and the stock code of the announcements. (ii) Regular expression: more event arguments can be matched to improve the recall of the labeled data through regular expression. for example, LONCIN CO LTD (Role=Shareholder Name) in the financial event database, but LONCIN in the announcement. We can solve this problem by regular expression and label the LONCIN as an event argument (iii) Rules: some task-driven rules can be used to automatically annotate data. for example, we can mark 12 months (Role=Pledging End Date) by calculating the date difference between 2017-02-23 (Role=Pledging Start Date) and 2018-02-23 (Role=Pledging End Date). Figure 4 depicts the overall architecture of the EE system proposed in this paper which primarily involves the following two components: The sentence-level Event Extraction (SEE) purposes to extract event arguments and event triggers from one sentence; The document-level Event Extraction (DEE) aims to extract event arguments from the whole document based on a key event detection model and an arguments-completion strategy.

## Event Extraction (EE)

## Sentence-level Event Extraction (SEE)

We formulate SEE as a sequence tagging task and the training data supported by sentence-level labeled data. Sentences are represented in the BIO format where each character (event triggers, event arguments and others) is labeled as B-label if the token is the beginning of an event argument, I-label if it is inside an event argument or Olabel if it is otherwise. In recent years, neural networks have been used in most NLP tasks because it can automatically learn features from the text representation. And the Bi-LSTM-CRF model can produce state of the art (or close to) accuracy on classic NLP tasks such as part of speech (POS), chunking and NER (Huang et al., 2015). It can effectively use both past and future input features thanks to a Bidirectional Long Short-Term Memory (BiLSTM) component and can also use sentence-level tag information thanks to a Conditional Random Field (CRF) layer.
The specific model implementation of the SEE, as shown on the left of Figure 4, is made up of a Bi-LSTM neural network and a CRF layer. Each Chinese character in a sentence is represented by a vector as the input of the Bi-LSTM layer 11 (Mikolov et al., 2013). The output of the Bi-LSTM layer is projected to score for each character. And a CRF layer is used to overcome the label-bias problem. The SEE eventually returns the result of the sentence-level EE for each sentence in the document.

## Document-level Event Extraction(DEE)

The DEE is composed of two parts: a key event detection model which aims to discover the event mention in the document and an argumentscompletion strategy which aims to pad the missing event arguments.
Key event detection: as shown on the right of figure 4, the input of the event detection is made up of two parts: one is the representation of the event arguments and event trigger come from the output of SEE (blue), and the other is the vector representation of the current sentence(red). The two parts are concatenated as the input feature of the Convolutional Neural Networks(CNN) layer. And then the current sentence is classified into two categories: a key event or not.
Arguments-completion strategy: We have obtained the key event which contains most of the event arguments by the DEE, and the event extraction results for each sentence in a document by the SEE. For obtaining complete event information, we use arguments-completion strategy which can automatically pad the missing event arguments from the surrounding sentences. As shown in figure 4, an integrated Pledge event contains event arguments in event mention S n and filled event argument 12 months obtained from the sentence S n+1 .

## Evaluation

## Dataset

We carry out experiments on four types of financial events: Equity Freeze (EF) event, Equity Pledge (EP) event, Equity Repurchase (ER) event and Equity Overweight (EO) event. A total of 2976 announcements have been labeled by automatically generating data. We divided the labeled data into three subsets: the training set (80% of the total number of announcements), development set (10%) and test set (10%). Table 1 shows the statistics of the dataset. NO.ANN represents the number of announcements can be labeled automatically for each event type. NO.POS represents the total number of positive case sentences (event mentions). On the contrary, NO.NEG represents the number of negative case sentences. The positive and negative case sentences constitute the document-level data as the training data for the DEE. The positive sentences which contain event trigger and a series of event arguments, are labeled as sentence-level training data for the SEE. We randomly select 200 samples (contain 862 event arguments) to manually evaluate the precision of the automatically labeled data. The average precision is shown in Table 2 which demonstrates that our automatically labeled data is of high quality.

## Performance of the System

We use the precision(P ), recall(R) and (F 1 ) to evaluate the DCFEE system. Table 3 shows the performance of the pattern-based method 12 and the DCFEE in the extraction of the Equity Freeze event. The experimental results show that the performance of the DCFEE is better than that of the pattern-based method in most event arguments extraction.    Table 4: P , R, F 1 of SEE, DEE on the different event types.
In conclusion, the experiments show that the method based on DS can automatically generate high-quality labeled data to avoid manually labeling. It also validates the DCFEE proposed in this paper, which can effectively extract events from the document-level view.

## Application of the DCFEE

The application of the DCFEE system: an online EE service for Chinese financial texts. It can help financial professionals quickly get the event information from the financial announcements. Figure 5 shows a screenshot of the online DCFEE system. Different color words represent different event arguments' types, underlined sentences represent the event mention in the document. As shown in figure 5, we can obtain a complete Equity Freeze event from unstructured text (an announcement about Equity Freeze).

## Related Work

The current EE approaches can be mainly classified into statistical methods, pattern-based method and hybrid method (Hogenboom et al., 2016).
Statistical method can be divided into two categories: traditional machine learning algorithm based on feature extraction engineering (Ahn, 2006), (Ji and Grishman, 2008), (Liao and Grishman, 2010), (Reichart and Barzilay, 2012) and neural network algorithm based on automatic feature extraction , (Nguyen et al., 2016), . The pattern method is usually used in industry because it can achieve higher accuracy, but meanwhile a lower recall. In order to improve recall, there are two main research directions: build relatively complete pattern library and use a semi-automatic method to build trigger dictionary , (Gu et al., 2016). Hybrid event-extraction methods combine statistical methods and pattern-based methods together (Jungermann and Morik, 2008), (Bjorne et al., 2010). To our best knowledge, there is no system that automatically generates labeled data, and extracts document-level events automatically from announcements in Chinese financial field.

## Conclusion

We present DCFEE, a framework which is able to extract document-level events from Chinese financial announcements based on automatically labeled data. The experimental results show the effectiveness of the system. We successfully put the system online and users can quickly get event information from the financial announcements through it9.

## Figure 2 :

## Figure 3 :

## Figure 4 :

## Figure 5 :

## Table 2 :

## Table 3 :

## Table 4

## Figures (text descriptions)

### Figure 1

Overview of the DCFEE framework.

Figure 2 :
2Overview of the DCFEE framework.

### Figure 2

The process of labeled data generation.

Figure 3 :
3The process of labeled data generation.

### Figure 3

11 Word vectors are trained with a version of word2vec on Chinese WiKi corpus The architecture of event extraction.

Figure 4 :
411 Word vectors are trained with a version of word2vec on Chinese WiKi corpus The architecture of event extraction.

### Figure 4

A screen shot of the online DCFEE system 9 .

Figure 5 :
5A screen shot of the online DCFEE system 9 .

### Figure 5

E1:The 520,000 shares held by Nagafu Ruihua were frozen by the people's Court of Dalian city in May 5, 2017. E2:The period of freezing is 3 years.

E1:The 520,000 shares held by Nagafu Ruihua were frozen by the people's Court of Dalian city in May 5, 2017. E2:The period of freezing is 3 years.C1: 
520,000 
2017 5 5 
C2: 
3 

Trigger 

Shareholder Name 

Number of Frozen Stock 

Frozen Institution 

Freezing Start Date 

Freezing End Date 
Trigger

### Figure 6

Dataset NO.ANN NO.POS NO.NEG

Dataset NO.ANN NO.POS NO.NEGEF 
526 
544 
2960 
EP 
752 
775 
6392 
EB 
1178 
1192 
11590 
EI 
520 
533 
11994 
Total 
2976 
3044 
32936 

Table 1: Statistics of automatically labeled data.

### Figure 7

Manual Evaluation Results.

Table 2 :
2Manual Evaluation Results.

### Figure 8

P , R, F 1 of pattern-based and DCFEE on the Equity Freeze event.

Table 3 :
3P , R, F 1 of pattern-based and DCFEE on the Equity Freeze event.

### Figure 9

shows the P , R, F 1 of SEE and DEE on the different event types. It is noteworthy that the golden data used in SEE stage is the automatically generated data and the golden data used in DEE stage comes from the financial event knowledge base. The experimental results show that the effectiveness of SEE and DEE, the acceptable precision 12 Example of a pattern for a freeze event: (Frozen institution(ORG)+, Trigger word(TRI)+, Shareholder names(NAME)+,time)    and expansibility of the DCFEE system presented in this paper.EO 88.76 91.88 90.25 80.77 45.93 58.56

Table 4
4shows the P , R, F 1 of SEE and DEE on the different event types. It is noteworthy that the golden data used in SEE stage is the automatically generated data and the golden data used in DEE stage comes from the financial event knowledge base. The experimental results show that the effectiveness of SEE and DEE, the acceptable precision 12 Example of a pattern for a freeze event: (Frozen institution(ORG)+, Trigger word(TRI)+, Shareholder names(NAME)+,time)    and expansibility of the DCFEE system presented in this paper.EO 88.76 91.88 90.25 80.77 45.93 58.56    Stage 
SEE 
DEE 
Type P (%) R(%) F1(%) P (%) R(%) F1(%) 
EF 
90.00 90.41 90.21 80.70 63.40 71.01 
EP 
93.31 94.36 93.84 80.36 65.91 72.30 
ER 
92.79 93.80 93.29 88.79 82.02 85.26

### Figure 10

Event Mention : Intermediate people's courthas of the Changchun city held a judicial freeze on the 43,070 million shares held by long group. …From August 12, 2016 to August 11, 2018.

Event Mention : Intermediate people's courthas of the Changchun city held a judicial freeze on the 43,070 million shares held by long group. …From August 12, 2016 to August 11, 2018.(Shareholder Name) 

(Frozen Institution) 

(Number of Frozen Stock) 

(Freezing Start Date) 

(Freezing End Date) 

(Trigger) 

(Equity Freeze) 

Overview 
Result 
Text 
Return

## References

1. The stages of event extraction. David Ahn, The Workshop on Annotating and Reasoning about Time and Events. David Ahn. 2006. The stages of event extraction. In The Workshop on Annotating and Reasoning about Time and Events, pages 1-8.

2. Complex event extraction at pubmed scale. Jari Bjorne, Filip Ginter, Sampo Pyysalo, Jun&apos;ichi Tsujii, Tapio Salakoski, Bioinformatics. 2612Jari Bjorne, Filip Ginter, Sampo Pyysalo, Jun'Ichi T- sujii, and Tapio Salakoski. 2010. Complex even- t extraction at pubmed scale. Bioinformatics, 26(12):i382-i390.

3. Automatically labeled data generation for large scale event extraction. Yubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, Proceedings of the ACL. the ACLYubo Chen, Shulin Liu, Xiang Zhang, Kang Liu, and Jun Zhao. 2017. Automatically labeled data genera- tion for large scale event extraction. In Proceedings of the ACL, pages 409-419.

4. Event extraction via dynamic multi-pooling convolutional neural networks. Yubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, Jun Zhao, Proceedings of the ACL. the ACLYubo Chen, Liheng Xu, Kang Liu, Daojian Zeng, and Jun Zhao. 2015. Event extraction via dynam- ic multi-pooling convolutional neural networks. In Proceedings of the ACL.

5. Incorporating copying mechanism in sequence-to-sequence learning. Jiatao Gu, Zhengdong Lu, Hang Li, O K Victor, Li, Proceedings of the ACL. the ACLJiatao Gu, Zhengdong Lu, Hang Li, and O.K. Vic- tor Li. 2016. Incorporating copying mechanism in sequence-to-sequence learning. In Proceedings of the ACL, pages 1631-1640.

6. A survey of event extraction methods from text for decision support systems. Frederik Hogenboom, Flavius Frasincar, Uzay Kaymak, Franciska De Jong, Emiel Caron, Decision Support Systems. 85Frederik Hogenboom, Flavius Frasincar, Uzay Kay- mak, Franciska De Jong, and Emiel Caron. 2016. A survey of event extraction methods from text for de- cision support systems. Decision Support Systems, 85(C):12-22.

7. Bidirectional lstm-crf models for sequence tagging. Zhiheng Huang, Wei Xu, Kai Yu, Computer ScienceZhiheng Huang, Wei Xu, and Kai Yu. 2015. Bidirec- tional lstm-crf models for sequence tagging. Com- puter Science.

8. Refining event extraction through cross-document inference. Heng Ji, Ralph Grishman, Proceedings of the ACL. the ACLHeng Ji and Ralph Grishman. 2008. Refining even- t extraction through cross-document inference. In Proceedings of the ACL, pages 254-262.

9. Enhanced Services for Targeted Information Retrieval by Event Extraction and Data Mining. Felix Jungermann, Katharina Morik, SpringerBerlin HeidelbergFelix Jungermann and Katharina Morik. 2008. En- hanced Services for Targeted Information Retrieval by Event Extraction and Data Mining. Springer Berlin Heidelberg.

10. Using document level cross-event inference to improve event extraction. Shasha Liao, Ralph Grishman, Proceedings of the ACL. the ACLShasha Liao and Ralph Grishman. 2010. Using doc- ument level cross-event inference to improve event extraction. In Proceedings of the ACL, pages 789- 797.

11. Exploiting argument information to improve event detection via supervised attention mechanisms. Shulin Liu, Yubo Chen, Kang Liu, Proceedings of the ACL. the ACLShulin Liu, Yubo Chen, Kang Liu, and Jun Zhao. 2017. Exploiting argument information to improve event detection via supervised attention mechanisms. In Proceedings of the ACL, pages 1789-1798.

12. Efficient estimation of word representations in vector space. Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean, Computer Science. Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient estimation of word represen- tations in vector space. Computer Science.

13. Joint event extraction via recurrent neural networks. Kyunghyun Thien Huu Nguyen, Ralph Cho, Grishman, Proceedings of the ACL. the ACLThien Huu Nguyen, Kyunghyun Cho, and Ralph Gr- ishman. 2016. Joint event extraction via recurrent neural networks. In Proceedings of the ACL, pages 300-309.

14. Multi event extraction guided by global constraints. Roi Reichart, Regina Barzilay, Proceedings of the NAACL. the NAACLRoi Reichart and Regina Barzilay. 2012. Multi event extraction guided by global constraints. In Proceed- ings of the NAACL, pages 70-79.

15. Distant supervision for relation extraction via piecewise convolutional neural networks. Daojian Zeng, Kang Liu, Yubo Chen, Jun Zhao, Proceedings of the EMNLP. the EMNLPDaojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao. 2015. Distant supervision for relation extraction via piecewise convolutional neural networks. In Pro- ceedings of the EMNLP, pages 1753-1762.
