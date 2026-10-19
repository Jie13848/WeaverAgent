# Progressive Boundary Refinement Network for Temporal Action Detection

```meta
corpus_id: 213307923
```

## Authors

- Qinying Liu 
Department of Automation
University of Science and Technology of China
- Zilei Wang zlwang@ustc.edu.cn 
Department of Automation
University of Science and Technology of China

## Abstract

Temporal action detection is a challenging task due to vagueness of action boundaries. To tackle this issue, we propose an end-to-end progressive boundary refinement network (PBR-Net) in this paper. PBRNet belongs to the family of one-stage detectors and is equipped with three cascaded detection modules for localizing action boundary more and more precisely. Specifically, PBRNet mainly consists of coarse pyramidal detection, refined pyramidal detection, and fine-grained detection. The first two modules build two feature pyramids to perform the anchor-based detection, and the third one explores the frame-level features to refine the boundaries of each action instance. In the fined-grained detection module, three frame-level classification branches are proposed to augment the frame-level features and update the confidence scores of action instances. Evidently, PBRNet integrates the anchorbased and frame-level methods. We experimentally evaluate the proposed PBRNet and comprehensively investigate the effect of the main components. The results show PBR-Net achieves the state-of-the-art detection performances on two popular benchmarks: THUMOS'14 and ActivityNet, and meanwhile possesses a high inference speed.

## Related Work

Most of existing temporal action detection methods can be categorized into two groups as follows.
The first group of methods (Caba Heilbron, Carlos Niebles, and Ghanem 2016; Shou, Wang, and Chang 2016;Buch et al. 2017b;Shou et al. 2017;Zhao et al. 2017;Qiu et al. 2018;Gao, Chen, and Nevatia 2018;Lin et al. 2018;Liu et al. 2019;Lin et al. 2019) adopt a two-stage scheme consisting of proposal and classification, which is fairly popular now. Among them, some works (Caba Heilbron, Carlos Niebles, and Ghanem 2016; Buch et al. 2017b;Gao, Chen, and Nevatia 2018;Lin et al. 2018;Liu et al. 2019;Lin et al. 2019) mainly focus on generating high-quality proposals, and a few works explore how to better classify the proposals Zhao et al. 2017;Yang et al. 2018). However, most of them train the two-stage networks separately. Recently, there are some attempts (Xu, Das, and Saenko 2017;Bai et al. 2018;Chao et al. 2018) to jointly train the twostage networks in a unified framework by following image object detector Faster R-CNN (Ren et al. 2015).
The second group of methods (Yeung et al. 2016;Buch et al. 2017a;Lin, Zhao, and Shou 2017a;Long et al. 2019) draw more attention on the one-stage architectures, motivated by the development of region-free image object detectors such as SSD (Liu et al. 2016). For example, SSAD (Lin, Zhao, and Shou 2017a) assigns anchors to the bottom-up pyramidal layers for predicting the classes and locations of actions. SS-TAD (Buch et al. 2017a) devises two recurrent memory modules to capture temporal information of videos, in which neither the feature pyramid nor anchor mechanism is adopted. Similarly, another work (Yeung et al. 2016) proposes a recurrent network with reinforcement learning. Recently, CTAN (Long et al. 2019) proposes to explore the temporal structure of an action through learning a Gaussian kernel for each cell in the pyramidal layers. Different from these works, our method pursues to build an end-to-end trainable network with anchor mechanism, aiming at taking advantages of two-stage methods into one-stage method.

## Progressive Boundary Refinement Network

The architecture of the proposed progressive boundary refinement network (PBRNet) is illustrated in Figure 1, where the U-net (Ronneberger, Fischer, and Brox 2015) like structure is particularly adopted as one important component. Especially, two types of fusion blocks (i.e., FBv1 and FBv2) are inserted to fuse features from different levels. PBR-Net consists of three key modules along with a spatiotemporal feature extractor, i.e., coarse pyramidal detection, refined pyramidal detection and fine-grained detection. Here the first two modules perform the anchor-based detection in multiple scales, and the last one completes fine-grained boundary refinement for each action candidate based on the frame-level feature. Features used in the three modules become much richer or finer gradually. In addition, the three modules will continuously refine action candidates produced by the previous module.

## Main Modules

Feature Extraction In this work, we segment videos into clips with a fixed temporal length as the inputs. Let (L × H × W ) denote the spatio-temporal shape of video clips, where L is the number of continuous frames, H and W are the spatial height and width of the video frames. In our experiments, H = W = 96 is set. For feature extraction, we employ I3D (Carreira and Zisserman 2017) as the visual encoder due to its excellent performance in action recognition (Carreira and Zisserman 2017) and temporal action detection (Chao et al. 2018). Specifically, we use the layers of I3D before the last average pooling layer as the backbone network for feature extraction. Then a feature map with the size (L/8 × H/32 × W/32) will be generated for each video clip.
Coarse Pyramidal Detection In this work, we build temporal feature pyramids in our anchor-based detection mod- ules to address the variation of action instances in different temporal scales. In CPD, we take the output of the backbone network as the first anchor layer. For other anchor layers, the temporal scale is reduced by a factor of 2 with maintaining the spatial shape of 3 × 3. Taking our implementation on THUMOS'14 (Jiang et al. 2014) for instance, the feature pyramid consists of five layers, denoted by (B 0 , B 1 , B 2 , B 3 , B 4 ), and the corresponding temporal scales are (L/8, L/16, L/32, L/64, L/128), respectively. Here the high-resolution feature maps in the low layers of the pyramid are designated to detect the short action instances and the low-resolution ones are for the long action instances. Like in SSD (Liu et al. 2016), we assign each layer of feature maps a set of preset anchors with several scales. Suppose a feature map has a temporal length of N , and we assign K anchors of different scales to each temporal location. Then there are NK anchors in total associated with this anchor layer. For each layer in the pyramid, we conduct the classification and location regression (i.e., predicting offsets to default boundaries) by 3D convolutional layers with the kernel size of (3 × 3 × 3). With CPD, we will obtain the first-level action candidates which are refined from preset anchors. However in the pyramid of CPD, the low layers lack enough semantics and the high layers lack enough fine details, hence the first-level action candidates only have coarse boundaries.

## Refined Pyramidal Detection

To amend the drawbacks of CPD, we propose a more powerful pyramid in RPD. The feature pyramid upscales the temporal resolution gradually, and enriches the features by merging with the feature maps of CPD, as shown in Figure 1. Thus, the feature hierarchy is (U 0 , U 1 , U 2 , U 3 , U 4 ) with the temporal scales (L/128, L/64, L/32, L/16, L/8) in our implementation on THUMOS'14. Specifically, the feature map U n in this module is generated by merging the feature map U n−1 in the previous layer and the corresponding feature map B m in CPD. More specifically, we use the first kind of fusion block  (i.e., FBv1) for feature fusion, as shown in Figure 2. In each FBv1, we use a 3D deconvolutional layer over U n−1 to enlarge the temporal length by a factor of 2, and then apply a 256-channel convolution on B m to fit the channels of U n−1 . The element-wise summation is followed to merge both features, and a convolutional layer is then applied to produce U n . Since CPD and RPD have symmetric structure, we directly assign the first-level action candidates as the anchors of RPD. Afterwards, we conduct the classification and regression to produce the second-level action candidates.
Although the structure of two inter-connected temporal pyramids is also applied in , the novelty of our method is that classification and regression are conducted on both pyramids to form a cascaded detection pipeline.
Fine-grained Detection FGD is designed to refine the candidates in a finer granularity. In FGD, we use the second type of fusion block (i.e., FBv2) to generate a framelevel feature. As shown in Figure 2, the frame-level feature is merged from two parts. The first part takes the last feature map of RPD as input, and then stacks three 3D deconvolutional layers to make temporal length of the feature map equal to the input clip. The second part takes the raw frames as input, and then uses three convolutional layers and a average pooling layer to make its spatio-temporal shape same with that of the first part. We concatenate these two feature maps for fusion, and then use a convolutional layer to generate the ultimate frame-level feature. The feature from the second part can supplement spatial details to the first part. Moreover, we add three frame-level classification branches upon the frame-level feature, which densely outputs three kinds of frame-level class-specific probabilities whether each frame is inside or outside, at or not at the boundaries of a ground-truth instance, denoted by actionness, starting and ending probability, respectively. Each branch is composed of a convolutional layer with the kernel size of (3 × 3 × 3) and a softmax layer. The frame-level classification branches have two main functions. First, the auxiliary frame-level supervisions can help to enrich the semantics of the frame-level feature, which will facilitate the fine-grained boundary regression. Second, the frame-level classification scores are used to fuse with the anchor-based scores in the inference time. For fine-grained boundary regression, we separately adjust the start and end time of each action candidate, as shown in Figure 3. Let s and e denote the start/end time of an action candidate, and we will equally process each of them. Without loss of generalization, we explain the refinement procedure by taking s for instance. To be specific, we first locate it in the frame-level feature map, and then choose the neighboring features centered at s as the input to refine s with the temporal length t/β (β = 8 is used in our experiments), where t = e−s is the length of the action candidate. We employ a temporally dilated 3D convolutional layer to predict the final boundary, where the network output is the temporal offset. The kernel size of the dilated convolutional layer is set as 3 × 3 × 3 and the temporal dilation rate is t/(2 · β). Such settings enable the receptive field of the dilated convolutional layer reach at t/β in the temporal dimension.
After the last time of boundary refinement, we get the third-level action candidates. Then we map each candidate on the frame-level classification branches, and find the corresponding class-specific starting probability of the start lo-cation and ending probability of the end location, denoted as p k s and p k e for the action class of k respectively. We will use these two frame-level class-specific scores to fuse with the anchor-based scores in the test phase.

## Training of PBRNet

Progressive Matching Strategy The matching strategy is to determine which anchors correspond to a ground-truth instance. In this work, we calculate the IoU scores of each anchor with all action instances. An anchor is regarded positive if its highest IoU score is greater than the preset threshold h, and background otherwise. For positive samples, the matched action instance with the highest score is used as the ground truth. We apply this matching strategy to three detection modules with the thresholds h cp , h rp and h fg , respectively. We set the IoU thresholds of three modules with increasing values such that actions can be refined progressively. In our implementation, h cp = 0.5, h rp = 0.6 and h fg = 0.7 are used. The similar idea is proposed by the multi-stage cascaded object detector (Cai and Vasconcelos 2018). Here we adopt it in our single-shot framework.
Preliminary Anchor Discarding There is serious imbalance between background and foreground after matching, which will badly bias the optimization. To alleviate this issue, we take two steps before feeding the anchors from one module into the next module. First, we discard some wellrecognized background anchors. Concretely, we compute the background score for each anchor according to the prediction of previous module, and then only keep the anchors whose background scores are lower than a constant threshold. Second, hard example mining is then applied to keep an acceptable balance between foreground and background samples. Specifically, we only preserve the background anchors with high loss values to make the amounts of the background and foreground anchors approximately equal.

## Loss Function

We use a multi-task loss function to train the network. The overall loss L total is defined as
where L cp , L rp , and L fg represent the loss functions for CPD, RPD and FGD, respectively. λ 1 and λ 2 are the control parameters which are both empirically set to 1.
Since the CPD and RPD have similar detection structures, L cp and L rp share a unified formulation L x as
where N is the number of training anchors, L cls is the classification loss, and L loc is the regression loss. Specifically, p i is the predicted classification scores for the anchor i and p * i is the corresponding ground-truth label. t = (t c , t l ) denotes the temporal offset of the predicted center location and duration, and t * is the corresponding ground truth.
L fg consists of two parts, i.e.,
The weight factor γ is empirically set as 1. L br fg represents the boundary regression loss, i.e.,
where N br is the number of training action candidates, b i = (s i , e i ) represents the offsets of the predicted start time and end time for the action candidate i, and b * i is the corresponding ground truth.
L fc fg is the loss function for frame-level classification, i.e.,
. where N f is the number of frames. p a i , p s i and p e i represent the predicted class-specific actionness probability, starting probability and ending probability of the frame i, respectively. The p a * i , p s * i and p e * i are the corresponding groundtruth label of the frame i. To assign frame-level label, for each ground-truth instance with boundary of (s * , e * ) and category of c * , we define the actionness region, starting region, ending region as [s * , e * ], [s * − t * /η, s * + t * /η] and [e * − t * /η, e * + t * /η], where t * = e * − s * is the duration of the ground-truth instance and η is empirically set as 10.
For each frame, its actionness, starting or ending label will be set as c * if it lies in the corresponding region.
We use the cross-entropy loss for classification and smooth 1 loss for regression throughout the experiments.

## Inference of PBRNet

Scores Fusion For each anchor, we first get two anchorbased classification scores output by CPD and RPD (denoted as p k cp and p k rp for action class k, respectively). We then fuse the anchor-based scores with the frame-level scores ( i.e., p k s and p k e ) from FGD by multiplication. Concretely, for each anchor the final confidence score of class k is computed as:
The score contain both the anchor-based information and fine-grained information, hence is able to help achieve better evaluation performances.

## Two-stream Fusion

In recent works on temporal action detection (Chao et al. 2018), fusing the appearance and optic flow information by two-stream architectures has shown significantly helpful to achieving good performance due to their complementarity. Similarly, we also investigate the two-stream fusion in our framework. Specifically, we train our proposed networks for two streams separately, and average the predicted values from both streams in the test phase.
Post-processing For each anchor, the fused confidence score and three-step regressed boundaries are used for evaluation. Non-maximum suppression (NMS) is conducted to reduce redundant results. We assemble the detection results of the clips belonging to the same video.

## Experiments Dataset and Setup

Dataset THUMOS'14 (Jiang et al. 2014) contains 200 untrimmed videos (including 3, 007 action instances) in the validation set and 213 untrimmed videos (including 3, 358 action instances) in the test set which are widely used for temporal action detection from 20 action categories. We use the validation set for training and the test set for evaluation. ActivityNet v1.3 (Caba Heilbron et al. 2015) contains 10, 024, 4, 926, and 5, 044 videos from 200 classes in the training, validation, and test sets, respectively. We evaluate our model on the validation set, as in previous works Gao, Chen, and Nevatia 2018;Xie et al. 2018).
Evaluation Metrics For evaluation, we report the mean average precision (mAP) using multiple IoU thresholds. On THUMOS'14, mAP is computed by its official toolkit (Jiang et al. 2014). On ActivityNet, following the protocol provided by the dataset ( Experimental Settings For training, adjacent clips are allowed to be temporally overlapped to increase the amount of training data. We only keep the clips containing at least one complete ground-truth action instance. In the inference stage, the input video is split into clips without temporal overlap. All clips are kept for evaluation. On THUMOS'14, we sample both RGB and optic flow frames at 10 frames per second (fps). The length of each clip L is set as 256 frames (i.e., about 25.6 seconds), which is greater than that of 99.7% of action instances in the dataset. Since Activi-tyNet is much larger-scale than THUMOS'14, we sample frames at only 3 fps on ActivityNet. Accordingly, L is set as 768 (i.e., covering 256 seconds of a video). Such a temporal length is longer than the duration of over 99.9% of action instances in the training set. We set 6 feature layers in the two pyramids. In addition, considering the fact that one video only contains a single class of action instances for most videos in ActivityNet, we first conduct binary classification in different modules, which indicates whether each anchor contains an action instance, and then assign each detected anchor the top-1 video-level label predicted by (Wang and Tao 2016), by following the protocol in (Lin et al. 2018;. Our backbone is pretrained by (Carreira and Zisserman 2017) on the ImageNet and Kinetics datasets. The batch size is set as 1, thus we freeze all batch normalization layers.  (Wang, Qiao, and Tang 2014) 14.6 12.1 8.5 4.7 1.5 FTP (Caba Heilbron, Carlos Niebles, and Ghanem 2016) − − 13.5 − − DAP (Escorcia et al. 2016) − − 13.9 − − Oneata et al. (Oneata, Verbeek, and Schmid 2014) 28.8 21.8 15.0 8.5 3.2 Richard and Gall (Richard and Gall 2016) 30.0 23.2 15.2 − − Yeung et al. (Yeung et al. 2016) 36.0 26.4 17.1 − − SMS (Yuan et al. 2016) 36.5 27.8 17.8 − − Yuan et al. (Yuan et al. 2017) 33.6 26.1 18.8 − − S-CNN (Shou, Wang, and Chang 2016) 36.3 28.7 19.0 10.3 5.3 SST (Buch et al. 2017b) 37.8 − 23.0 − − CDC  40.1 29.4 23.3 13.1 7.9 SSAD (Lin, Zhao, and Shou 2017a) 43.0 35.0 24.6 − − TCN (Dai et al. 2017) − 33.3 25.6 15.9 9.0 TURN  44.1 34.9 25.6 − − Xiong et al.  48.7 39.8 28.2 − − R-C3D (Xu, Das, and Saenko 2017) 44.8 35.6 28.9 − − SS-TAD (Buch et al. 2017a) 45.7 − 29.2 − 9.6 SSN  51.9 41.0 29.8 − − CTAP (Gao, Chen, and Nevatia 2018) − − 29.9 − − CBR  50.1 41.3 31.0 19.1 9.9 ETP (Qiu et al. 2018) 48  great potential to beat the two-stage detectors.

## Action Detection Performance

## Inference Speed

We present the comparison of action detection speed between our PBRNet and other state-of-art methods, as shown in Table 3. Here our model is evaluated on a Nvidia GeFore GTX 1080Ti GPU. It can be seen that PBRNet is able to process frames at a speed of 1488 fps, and thus is one of most competitive methods. The high efficiency of our PBR-Net may mainly from the following two facts. First, PBRNet belongs to the family of one-stage detectors and thus eliminates the additional proposal stage used in S-CNN, DAP and R-C3D. Second, PBRNet adopts the fully convolutional operations to process frames rather than the time-consuming recurrent architectures used in DAP and SS-TAD.

## Ablation Study

In order to investigate the effect of key components and settings in our proposed PBRNet, we conduct ablation study on THUMOS'14. Here we only use the threshold of 0.5 to evaluate the methods without specification, since different thresholds have similar performance trends. The results of RGB stream and optical stream are reported separately.

## Main Detection Modules

Anchors are progressively revised in three modules. Here we investigate the performances after using different numbers of modules, where the multiplication is applied to fuse the confidence scores from all used modules. The results are shown in Table 4. It can be seen that the detection performance is boosted gradually as higher-level modules are used to refine action anchors, validating the progressive design of our PBRNet.

## Frame-level Classification Branches

In FGD, three frame-level classification branches are used to enrich the frame-level feature and generate ultimate confidence score.
Here the performance are evaluated when the frame-level classification branches are all removed. The second row in Table 5 shows the frame-level classification branches are helpful to boost the action detection performances.
Progressive Matching Strategy We adopt the increasing IoU thresholds in PBRNet to match action anchors in the detection modules. Here we conduct another experiment with the same IoU threshold of 0.5 as a comparison. As shown in tIoU 0.5 0.75 0.95 Average R-C3D (Xu, Das, and Saenko 2017) 26.80 − − 12.70 CMS-RC3D (Bai et al. 2018) 32.92 18.36 1.13 18.46 TCN (Dai et al. 2017) 36.17 21.12 3.89 − TAL-Net (Chao et al. 2018) 38.23 18.30 1.30 20.22 CDC  43.83 25.88 0.21 22.77 Xiong et al.  39.12 23.48 5.49 23.98 Lin et al. (Lin, Zhao, and Shou 2017b)      the third row of Table 5, the proposed setting is positive to the performance of action detection.
Preliminary Anchor Discarding This is used to filter out the background anchors with high confidence scores for subsequent detection modules, before applying hard example mining. We also conduct an ablation experiment where only hard example mining is used. The results in the fourth row of Table 5 show that the proposed strategy is useful to our network.

## Conclusion

In this paper, we proposed a novel progressive boundary refinement network (PBRNet) for temporal action detection, which progressively refine action anchors under the onestage detection framework. Particularly, we introduced two inter-connected pyramids to perform the anchor-based detection and one refinement module to accurately localize the action boundaries in a fine granularity. Benefited from the well-designed architecture, end-to-end training, and specific learning methods, PBRNet achieves the state-of-the-art performance on THUMOS'14 and ActivityNet.

## Figure 1 :

## Figure 2 :

## Figure 3 :

## Figure 4 :

## Table 1

## Table 1 :

## Table 2

## Table 2 :

## Table 3 :

## Table 4 :

## Table 5 :

## Figures (text descriptions)

### Figure 1

Architecture of our proposed progressive boundary refinement network (PBRNet). PBRNet belongs to the one-stage detectors and mainly consists of three key components: coarse pyramidal detection, refined pyramidal detection and fine-grained detection. The three components sequentially refine the boundaries of action candidates accompanied by the enhancement of features. Best viewed in color.

Figure 1 :
1Architecture of our proposed progressive boundary refinement network (PBRNet). PBRNet belongs to the one-stage detectors and mainly consists of three key components: coarse pyramidal detection, refined pyramidal detection and fine-grained detection. The three components sequentially refine the boundaries of action candidates accompanied by the enhancement of features. Best viewed in color.

### Figure 2

Illustration of two kinds of fusion blocks. (a) FBv1, (b) FBv2. The kernel sizes (along with channels if necessary) and strides of operators are shown. The strides are (1,1,1) where not specified. Best viewed in color.

Figure 2 :
2Illustration of two kinds of fusion blocks. (a) FBv1, (b) FBv2. The kernel sizes (along with channels if necessary) and strides of operators are shown. The strides are (1,1,1) where not specified. Best viewed in color.

### Figure 3

Illustration of boundary refinement in FGD. Best viewed in color.

Figure 3 :
3Illustration of boundary refinement in FGD. Best viewed in color.

### Figure 4

Caba Heilbron et al. 2015), the IoU thresholds are set as [0.5 : 0.05 : 0.95].

Caba Heilbron et al. 2015), the IoU thresholds are set as [0.5 : 0.05 : 0.95].

### Figure 5

Qualitative examples from THUMOS'14 and Ac-tivityNet. Here we compare the predictions from different levels with ground truth. All the temporal boundaries are showed in seconds. Best viewed in color. Methods FPS S-CNN (Shou, Wang, and Chang 2016) 60 DAP (Escorcia et al. 2016) 134 CDC (Shou et al. 2017) 500 SS-TAD (Buch et al. 2017a) (Titan Xm) 701 R-C3D (Xu, Das, and Saenko 2017) (Titan Xm) 569 R-C3D (Xu, Das, and Saenko 2017) (Titan Xp) 1030 PBRNet (1080Ti) 1488

Figure 4 :
4Qualitative examples from THUMOS'14 and Ac-tivityNet. Here we compare the predictions from different levels with ground truth. All the temporal boundaries are showed in seconds. Best viewed in color. Methods FPS S-CNN (Shou, Wang, and Chang 2016) 60 DAP (Escorcia et al. 2016) 134 CDC (Shou et al. 2017) 500 SS-TAD (Buch et al. 2017a) (Titan Xm) 701 R-C3D (Xu, Das, and Saenko 2017) (Titan Xm) 569 R-C3D (Xu, Das, and Saenko 2017) (Titan Xp) 1030 PBRNet (1080Ti) 1488

### Figure 6

Temporal action detection mAP (%) on THUMOS'14.

Table 1
1gives the detection performance comparison of our 
proposed PBRNet with state-of-the-art methods, where the 
results after two-stream fusion are reported. It can be seen 
that PBRNet outperforms previous state-of-the-art methods 
with remarkable mAP gaps for all used IoU thresholds. 
Particularly, PBRNet achieves the improvement of 8.5% at 
tIoU = 0.5 (from 42.8% to 51.3%) compared with TAL-
Net, which is the state-of-the-art two-stage detector in THU-
MOS'14. Such results imply that one-stage detectors have

### Figure 7

gives the action detection results on ActivityNet. Similarly, PBR-Net surpasses other state-of-the-art methods for all listed IoU thresholds, although we sample frames of ActivityNet at only 3 fps. Some qualitative examples are shown in Figure 4.

Table 1 :
1Temporal action detection mAP (%) on THUMOS'14.

### Figure 8

Temporal action detection mAP (%) on ActivityNet v1.3 (val)

Table 2
2gives the action detection results on ActivityNet. Similarly, PBR-Net surpasses other state-of-the-art methods for all listed IoU thresholds, although we sample frames of ActivityNet at only 3 fps. Some qualitative examples are shown in Figure 4.

### Figure 9

Comparison on action detection speed for test.

Table 2 :
2Temporal action detection mAP (%) on ActivityNet v1.3 (val)22.8 
27.0 

24.8 
27.6 
33.9 
38.1 

LongJump 

23.7 

23.0 

26.9 

26.9 

LongJump 
31.7 

37.5 
31.1 

31.1 
37.2 

Ground-truth 
First-level 
Candidate 

Second-level 
Candidate 

Third-level 
Candidate 

37.0 

12.9 
27.1 

Platform diving 
8.2 

29.5 
11.1 

31.5 

30.2 
9.8

### Figure 10

Ablation study of different modules in PBRNet.

Table 3 :
3Comparison on action detection speed for test.Module 
mAP@0.5 
CPD RPD FGD RGB Flow 
36.8 37.3 
39.7 40.6 
42.2 42.4

### Figure 11

Techniques in PBRNet mAP@0.5 RGB Flow w/o Frame-level classification branches 41.2 40.3

Table 4 :
4Ablation study of different modules in PBRNet.

### Figure 12

Effect of different techniques in PBRNet, where the last row of results is the baseline.

Techniques in PBRNet mAP@0.5 RGB Flow w/o Frame-level classification branches 41.2 40.3w/o Progressive matching strategy 
41.3 41.5 
w/o Preliminary anchor discarding 
42.0 41.7 
full 
42.2 42.4

## References

1. Diagnosing error in temporal action detectors. H Alwassel, F Caba Heilbron, V Escorcia, B Ghanem, ECCV. Alwassel, H.; Caba Heilbron, F.; Escorcia, V.; and Ghanem, B. 2018. Diagnosing error in temporal action detectors. In ECCV.

2. Contextual multi-scale region convolutional 3d network for activity detection. Y Bai, H Xu, K Saenko, B Ghanem, arXiv:1801.09184arXiv preprintBai, Y.; Xu, H.; Saenko, K.; and Ghanem, B. 2018. Contextual multi-scale region convolutional 3d network for activity detection. arXiv preprint arXiv:1801.09184. (DOI: arXiv:1801.09184)

3. End-to-end, single-stream temporal action detection in untrimmed videos. S Buch, V Escorcia, B Ghanem, L Fei-Fei, J Niebles, BMVC. Buch, S.; Escorcia, V.; Ghanem, B.; Fei-Fei, L.; and Niebles, J. 2017a. End-to-end, single-stream temporal action detection in untrimmed videos. In BMVC.

4. Sst: Single-stream temporal action proposals. S Buch, V Escorcia, C Shen, B Ghanem, J C Niebles, CVPR. Buch, S.; Escorcia, V.; Shen, C.; Ghanem, B.; and Niebles, J. C. 2017b. Sst: Single-stream temporal action proposals. In CVPR.

5. Activitynet: A large-scale video benchmark for human activity understanding. F Caba Heilbron, V Escorcia, B Ghanem, Carlos Niebles, J , CVPR. Caba Heilbron, F.; Escorcia, V.; Ghanem, B.; and Carlos Niebles, J. 2015. Activitynet: A large-scale video benchmark for human activity understanding. In CVPR.

6. Fast temporal activity proposals for efficient detection of human actions in untrimmed videos. F Caba Heilbron, J Carlos Niebles, B Ghanem, CVPR. Caba Heilbron, F.; Carlos Niebles, J.; and Ghanem, B. 2016. Fast temporal activity proposals for efficient detection of human actions in untrimmed videos. In CVPR.

7. Cascade r-cnn: Delving into high quality object detection. Z Cai, N Vasconcelos, CVPR. Cai, Z., and Vasconcelos, N. 2018. Cascade r-cnn: Delving into high quality object detection. In CVPR.

8. Quo vadis, action recognition? a new model and the kinetics dataset. J Carreira, A Zisserman, CVPR. Carreira, J., and Zisserman, A. 2017. Quo vadis, action recogni- tion? a new model and the kinetics dataset. In CVPR.

9. Rethinking the faster r-cnn architecture for temporal action localization. Y.-W Chao, S Vijayanarasimhan, B Seybold, D A Ross, J Deng, R Sukthankar, CVPR. Chao, Y.-W.; Vijayanarasimhan, S.; Seybold, B.; Ross, D. A.; Deng, J.; and Sukthankar, R. 2018. Rethinking the faster r-cnn architecture for temporal action localization. In CVPR.

10. Temporal context network for activity localization in videos. X Dai, B Singh, G Zhang, L S Davis, Y Q Chen, ICCV. Dai, X.; Singh, B.; Zhang, G.; Davis, L. S.; and Chen, Y. Q. 2017. Temporal context network for activity localization in videos. In ICCV.

11. Daps: Deep action proposals for action understanding. V Escorcia, F C Heilbron, J C Niebles, B Ghanem, ECCV. Escorcia, V.; Heilbron, F. C.; Niebles, J. C.; and Ghanem, B. 2016. Daps: Deep action proposals for action understanding. In ECCV.

12. Turn tap: Temporal unit regression network for temporal action proposals. J Gao, Z Yang, C Sun, K Chen, R Nevatia, ICCV. Gao, J.; Yang, Z.; Sun, C.; Chen, K.; and Nevatia, R. 2017. Turn tap: Temporal unit regression network for temporal action propos- als. In ICCV.

13. Ctap: Complementary temporal action proposal generation. J Gao, K Chen, R Nevatia, ECCV. Gao, J.; Chen, K.; and Nevatia, R. 2018. Ctap: Complementary temporal action proposal generation. In ECCV.

14. Cascaded boundary regression for temporal action detection. J Gao, Z Yang, R Nevatia, BMVC. Gao, J.; Yang, Z.; and Nevatia, R. 2017. Cascaded boundary re- gression for temporal action detection. In BMVC.

15. Actor and action video segmentation from a sentence. K Gavrilyuk, A Ghodrati, Z Li, C G Snoek, CVPR. Gavrilyuk, K.; Ghodrati, A.; Li, Z.; and Snoek, C. G. 2018. Actor and action video segmentation from a sentence. In CVPR.

16. Thumos challenge: Action recognition with a large number of classes. Y Jiang, J Liu, A R Zamir, G Toderici, I Laptev, M Shah, R Sukthankar, Jiang, Y.; Liu, J.; Zamir, A. R.; Toderici, G.; Laptev, I.; Shah, M.; and Sukthankar, R. 2014. Thumos challenge: Action recognition with a large number of classes.

17. Deep learning. Y Lecun, Y Bengio, G Hinton, nature. 5217553436LeCun, Y.; Bengio, Y.; and Hinton, G. 2015. Deep learning. nature 521(7553):436.

18. Bsn: Boundary sensitive network for temporal action proposal generation. T Lin, X Zhao, H Su, C Wang, Yang , M , ECCV. Lin, T.; Zhao, X.; Su, H.; Wang, C.; and Yang, M. 2018. Bsn: Boundary sensitive network for temporal action proposal genera- tion. In ECCV.

19. Bmn: Boundary-matching network for temporal action proposal generation. T Lin, X Liu, X Li, E Ding, S Wen, arXiv:1907.09702arXiv preprintLin, T.; Liu, X.; Li, X.; Ding, E.; and Wen, S. 2019. Bmn: Boundary-matching network for temporal action proposal gener- ation. arXiv preprint arXiv:1907.09702. (DOI: arXiv:1907.09702)

20. Single shot temporal action detection. T Lin, X Zhao, Z Shou, ACMMM. Lin, T.; Zhao, X.; and Shou, Z. 2017a. Single shot temporal action detection. In ACMMM.

21. Temporal convolution based action proposal: Submission to activitynet. T Lin, X Zhao, Z Shou, arXiv:1707.06750arXiv preprintLin, T.; Zhao, X.; and Shou, Z. 2017b. Temporal convolution based action proposal: Submission to activitynet 2017. arXiv preprint arXiv:1707.06750. (DOI: arXiv:1707.06750)

22. Ssd: Single shot multibox detector. W Liu, D Anguelov, D Erhan, C Szegedy, S Reed, C.-Y Fu, A C Berg, ECCV. Liu, W.; Anguelov, D.; Erhan, D.; Szegedy, C.; Reed, S.; Fu, C.- Y.; and Berg, A. C. 2016. Ssd: Single shot multibox detector. In ECCV.

23. Multigranularity generator for temporal action proposal. Y Liu, L Ma, Y Zhang, W Liu, S.-F Chang, CVPR. Liu, Y.; Ma, L.; Zhang, Y.; Liu, W.; and Chang, S.-F. 2019. Multi- granularity generator for temporal action proposal. In CVPR.

24. Gaussian temporal awareness networks for action localization. F Long, T Yao, Z Qiu, X Tian, J Luo, T Mei, CVPR. Long, F.; Yao, T.; Qiu, Z.; Tian, X.; Luo, J.; and Mei, T. 2019. Gaussian temporal awareness networks for action localization. In CVPR.

25. The lear submission at thumos 2014. THUMOS14 Action Recognition Challenge. D Oneata, J Verbeek, C Schmid, Oneata, D.; Verbeek, J.; and Schmid, C. 2014. The lear submission at thumos 2014. THUMOS14 Action Recognition Challenge.

26. Precise temporal action localization by evolving temporal proposals. H Qiu, Y Zheng, H Ye, Y Lu, F Wang, L He, ICMR. Qiu, H.; Zheng, Y.; Ye, H.; Lu, Y.; Wang, F.; and He, L. 2018. Pre- cise temporal action localization by evolving temporal proposals. In ICMR.

27. Faster r-cnn: Towards real-time object detection with region proposal networks. S Ren, K He, R Girshick, J Sun, NIPS. Ren, S.; He, K.; Girshick, R.; and Sun, J. 2015. Faster r-cnn: Towards real-time object detection with region proposal networks. In NIPS.

28. Temporal action detection using a statistical language model. A Richard, J Gall, CVPR. Richard, A., and Gall, J. 2016. Temporal action detection using a statistical language model. In CVPR.

29. U-net: Convolutional networks for biomedical image segmentation. O Ronneberger, P Fischer, T Brox, MICCAI. Ronneberger, O.; Fischer, P.; and Brox, T. 2015. U-net: Convolu- tional networks for biomedical image segmentation. In MICCAI.

30. Cdc: Convolutional-de-convolutional networks for precise temporal action localization in untrimmed videos. Z Shou, J Chan, A Zareian, K Miyazawa, S.-F Chang, CVPR. CVPR. Shou, Z.Wang, D.and Chang, S.-F.Temporal action localization in untrimmed videos via multi-stage cnnsShou, Z.; Chan, J.; Zareian, A.; Miyazawa, K.; and Chang, S.-F. 2017. Cdc: Convolutional-de-convolutional networks for precise temporal action localization in untrimmed videos. In CVPR. Shou, Z.; Wang, D.; and Chang, S.-F. 2016. Temporal action lo- calization in untrimmed videos via multi-stage cnns. In CVPR.

31. Uts at activitynet 2016. R Wang, D Tao, AcitivityNet Large Scale Activity Recognition Challenge. 8Wang, R., and Tao, D. 2016. Uts at activitynet 2016. AcitivityNet Large Scale Activity Recognition Challenge 2016:8.

32. Action recognition and detection by combining motion and appearance features. L Wang, Y Qiao, X Tang, THU-MOS14 Action Recognition Challenge. 122Wang, L.; Qiao, Y.; and Tang, X. 2014. Action recognition and detection by combining motion and appearance features. THU- MOS14 Action Recognition Challenge 1(2):2.

33. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. S Xie, C Sun, J Huang, Z Tu, K Murphy, ECCV. Xie, S.; Sun, C.; Huang, J.; Tu, Z.; and Murphy, K. 2018. Rethink- ing spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In ECCV.

34. A pursuit of temporal accuracy in general activity detection. Y Xiong, Y Zhao, L Wang, D Lin, X Tang, CVPR. Xiong, Y.; Zhao, Y.; Wang, L.; Lin, D.; and Tang, X. 2017. A pur- suit of temporal accuracy in general activity detection. In CVPR.

35. R-c3d: region convolutional 3d network for temporal activity detection. H Xu, A Das, K Saenko, ICCV. Xu, H.; Das, A.; and Saenko, K. 2017. R-c3d: region convolutional 3d network for temporal activity detection. In ICCV.

36. Weakly supervised actor-action segmentation via robust multi-task ranking. Y Yan, C Xu, D Cai, J Corso, languageYan, Y.; Xu, C.; Cai, D.; and Corso, J. 2017. Weakly supervised actor-action segmentation via robust multi-task ranking. language.

37. Exploring temporal preservation networks for precise temporal action localization. K Yang, P Qiao, D Li, S Lv, Y Dou, AAAI. Yang, K.; Qiao, P.; Li, D.; Lv, S.; and Dou, Y. 2018. Exploring temporal preservation networks for precise temporal action local- ization. In AAAI.

38. Endto-end learning of action detection from frame glimpses in videos. S Yeung, O Russakovsky, G Mori, L Fei-Fei, CVPR. Yeung, S.; Russakovsky, O.; Mori, G.; and Fei-Fei, L. 2016. End- to-end learning of action detection from frame glimpses in videos. In CVPR.

39. Temporal action localization with pyramid of score distribution features. J Yuan, B Ni, X Yang, A A Kassim, CVPR. Yuan, J.; Ni, B.; Yang, X.; and Kassim, A. A. 2016. Temporal action localization with pyramid of score distribution features. In CVPR.

40. Temporal action localization by structured maximal sums. Z.-H Yuan, J C Stroud, T Lu, J Deng, CVPR. Yuan, Z.-H.; Stroud, J. C.; Lu, T.; and Deng, J. 2017. Temporal action localization by structured maximal sums. In CVPR.

41. Temporal action detection with structured segment networks. Y Zhao, Y Xiong, L Wang, Z Wu, X Tang, D Lin, ICCV. Zhao, Y.; Xiong, Y.; Wang, L.; Wu, Z.; Tang, X.; and Lin, D. 2017. Temporal action detection with structured segment networks. In ICCV.

42. Mict: Mixed 3d/2d convolutional tube for human action recognition. Y Zhou, X Sun, Z.-J Zha, W Zeng, CVPR. Zhou, Y.; Sun, X.; Zha, Z.-J.; and Zeng, W. 2018. Mict: Mixed 3d/2d convolutional tube for human action recognition. In CVPR.
