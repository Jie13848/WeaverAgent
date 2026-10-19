# Dynamically localizing multiple speakers based on the time-frequency domain

```meta
corpus_id: 233188165
```

## Authors

- Hodaya Hammer 
Faculty of Electrical Engineering
Ramat-GanIsrael
- Shlomo E Chazan 
Faculty of Electrical Engineering
Ramat-GanIsrael
- Jacob Goldberger 
Faculty of Electrical Engineering
Ramat-GanIsrael
- Sharon Gannot sharon.gannot@biu.ac.il 
Faculty of Electrical Engineering
Ramat-GanIsrael

## Abstract

In this study, we present a deep neural network-based online multi-speaker localization algorithm based on a multi-microphone array.Following the W-disjoint orthogonality principle in the spectral domain, time-frequency (TF) bin is dominated by a single speaker and hence by a single direction of arrival (DOA).A fully convolutional network is trained with instantaneous spatial features to estimate the DOA for each TF bin.The high-resolution classification enables the network to accurately and simultaneously localize and track multiple speakers, both static and dynamic.Elaborated experimental study using simulated and real-life recordings in static and dynamic scenarios demonstrates that the proposed algorithm significantly outperforms both classic and recent deep-learning-based algorithms.Finally, as a byproduct, we further show that the proposed method is also capable of separating moving speakers by the application of the obtained TF masks.

## Introduction

Localizing multiple sound sources recorded with a microphone array in an acoustic environment is an essential component in various cases such as source separation and scene analysis.The relative location of a sound source with respect to a microphone array is specified in the term of the DOA of the sound wave originating from that location.DOA estimation and tracking are essential building blocks in all modern far-field speech enhancement and recognition for smart home devices as well as robot audition applications.In real-life environments, sound sources are captured by the microphones together with acoustic reverberation.While propagating in an acoustic enclosure, the sound wave undergoes reflections from the room facets and from various objects.These reflections deteriorate speech quality and, in extreme cases, its intelligibility.Furthermore, reverberation increases the time dependency between speech frames, making source DOA estimation a very challenging task.
A plethora of classic signal processing-based approaches have been proposed throughout the years for the task of broadband DOA estimation.The multiple signal classification (MUSIC) algorithm [1] applies a subspace method that was later adapted to the challenges of speech processing in [2].The steered response power with phase transform (SRP-PHAT) algorithm [3] used a generalization of cross-correlation methods for DOA estimation.These methods are still widely in use for both single-and multi-speaker localization tasks.However, in highly reverberant enclosures, their performance rapidly deteriorates [4,5].
Supervised learning methods can be potentially advantageous for this task since they are data-driven.Deep neural networks can be trained to find the DOA in different acoustic conditions.Moreover, if a network is trained using rooms with different acoustic conditions and multiple noise types, it can be made robust against noise and reverberation even for rooms which were not in the training set.Deep learning methods have recently been proposed for sound source localization.In [6,7], simple feed-forward deep neural networks (DNNs) were trained using generalized cross correlation (GCC)-based audio features, demonstrating improved performance as compared with classical approaches.Yet, this method is mainly designed to deal with a single sound source at a time.An extension of the multi-speaker DOA, using a DNN for the estimation task, can be found in [8].In high reverberation conditions, however, the performance of these algorithms is not satisfactory.In [9] and [10], timedomain features were used demonstrating performance improvement in highly reverberant enclosures.In [11], a CNN-based classification method was applied in the short-time Fourier transform (STFT) domain for broadband DOA estimation, assuming that only a single speaker is active per time frame.The phase component of the STFT coefficients of the input signal were directly provided as input to the CNN.This work was extended by [5] to estimate multiple speakers' DOAs and has shown high DOA classification performance.
The main drawback of most DNN-based approaches, however, is that they only use low-resolution supervision, namely at the time frame level or even utterance-based level, and the network outputs a single localization decision for the entire time frame.For speech signals, however, each time-frequency bin is dominated by a different speaker, a property referred to as W-disjoint orthogonality (WDO) [12].In the case of multiple speakers, each TF bin can therefore be associated with a different DOA.This high-resolution information can yield an improved DOA estimation also in the entire time frame localization resolution, especially in the case of multiple speakers.
In this study, we present a multi-speaker DOA estimation algorithm that is based on the U-net architecture that infers the DOA of each TF bin.The DOA decisions of all the frequency bands of a single time frame are then aggregated to extract the active speakers at that time frame level.The TF-based classification also facilitates the tracking capabilities of multiple moving speakers.U-Net has been introduced in the medical imaging domain [13] and was recently successfully applied to various audio processing tasks, e.g., for speech dereverberation [14], speaker separation [15], and noise reduction [16], all in the STFT domain, and for speech enhancement in the time-domain [17,18] also employing self-attention mechanism.
In the current study, we show that U-net architecture is also beneficial in speaker localization and tracking applications.We tested the proposed method on simulated data, using publicly available room impulse responses (RIRs) recorded in a real room [19], as well as real-life experiments recorded at the acoustic lab, Bar-Ilan University.We show that the proposed algorithm significantly outperforms state-of-the-art methods.
The main contribution of our work is casting the time-domain DOA estimation problem into a timefrequency segmentation problem.The proposed method improves the DOA estimation performance with respect to (w.r.t.) the state-of-the-art (SOTA) approaches, which are frame-based, and facilitates simultaneous tracking of multiple moving speakers.

## Multiple-speaker localization algorithm

In this section, we describe the proposed algorithm, including the feature extraction, the network architecture, and the training procedure.

## Multi-microphone time-frequency features

Consider an array with M microphones acquiring a mixture of N speech sources in a reverberant environment.The ith speech signal s i (t) propagates through the acoustic channel before being acquired by the mth microphone:
where h i m is the RIR relating the ith speaker and the mth microphone.In the STFT domain, (1) can be written as (provided that the frame-length is sufficiently large w.r.t. the filter length):
where l and k are the time frame and the frequency indices, respectively.The STFT (2) is complex-valued and hence comprises both magnitude and phase information.It is clear that the magnitude information alone is insufficient for DOA estimation.It is therefore a common practice to use the phase of the TF representation of the received microphone signals, or their respective phase-difference, as they are directly related to the DOA in non-reverberant environments.We decided to use an alternative feature, which is generally independent of the speech signal and is mainly determined by the spatial information.For that, we have selected the relative transfer function (RTF) [20] as our feature, since it is known to encapsulate the spatial fingerprint for each sound source.Specifically, we use the instantaneous relative transfer function (iRTF), which is the bin-wise ratio between the mth microphone signal and the reference microphone signal z ref (l, k):
Note that the reference microphone is arbitrarily chosen.Reference microphone selection is beyond the scope of this paper (see [21] for a reference microphone selection method).The input feature set extracted from the recorded signal is thus a 3D tensor R:
The tensor R is constructed from L × K bins, where L is the number of time frames and K is the number of frequencies.Since the iRTFs are normalized by the reference microphone, the latter is excluded from the features.Then, for each TF bin (l, k), there are P = 2(M − 1) channels, where the multiplication by 2 is due to the real and imaginary parts of the complex-valued feature.For each TF bin, the spatial features were normalized to have a zero mean and a unit variance.Other feature extraction methods can be considered.In Section 3, we show that the features described above are a suitable choice for the localization task.

## U-Net for DOA estimation

The WDO assumption [12,22] implies that each TF bin (l, k) is dominated by a single speaker.Consequently, as the speakers are spatially separated, i.e., located at different DOAs, each TF bin is dominated by a single DOA.We first accurately estimate the speaker direction at every TF bin from the given mixed recorded signal.Then, we extract the speakers' locations at each time frame.We formulated the DOA estimation as a classification task by discretizing the DOA range.The resolution was set to 5 • , such that the DOA candidates are in the set
It is natural to view the DOA estimation as a regression problem.The regression output is a Gaussian unimodal distribution.Casting the problem as a classification yields a multi-modal distribution which is more suitable for the case of several speakers.Let D l,k be a random variable (r.v.) representing the active dominant direction, recorded at bin (l, k).Our task boils down to deducing the conditional distribution of the discrete set of DOAs in for each TF bin, given the recorded mixed signal:
For this task, we use a DNN.The network output is an
. Under this construction of the feature tensor and output probability tensor, a pixel-to-pixel approach [23] for mapping a 3D input "image, " R, and a 3D output "image, " P, can be utilized.A U-net is used to compute (5) for each TF bin.The pixel-to-pixel method is beneficial in two ways.First, for each TF bin in our input image, the network estimates the DOA distribution separately.Second, the TF supervision is carried out with the spectrum of the different speakers.The U-Net hence takes advantage of the spectral structure and the continuity of the sound sources in both the time and frequency axes.These structures contribute to the pixel-wise classification task and prevent discontinuity in the DOA decisions over time.In our implementation, we used a U-net architecture, similar to the one described in [24].
The input to the network is the feature tensor R (see ( 4)).In our U-net architecture, the input shape is (L, K, P) where K = 256 is the number of frequency bins, L = 256 is the number of frames, and P = 2M − 2 where M is the number of microphones.
TF bins in which there is no active speech are noninformative.Therefore, the estimation is carried out only on speech-active TF bins.As we assume that the acquired signals are noiseless, we define a TF-based voice activity detector (VAD) as follows:
where is a threshold value.In noisy scenarios, we can use a robust speech presence probability (SPP) estimator instead [25].
The DOAs should only be estimated on a time frame basis.Hence, we aggregate over all active frequencies at time frame l to obtain a frame-wise probability:
where K is the number of frequency bands for which (6) exceed the threshold at the lth time frame.We thus obtain for each time frame a posterior distribution over all possible DOAs.If the number of speakers is known in advance, we can choose the directions corresponding to the highest posterior probabilities.If an estimate of the number of speakers is also required, it can be determined by applying a proper threshold.We dub our algorithm timefrequency direction-of-arrival net (TF-DOAnet).Figure 1 summarizes the TF-DOAnet network architecture.The algorithm is summarized in Table 1.

## Model training

The supervision in the training phase is based on the WDO assumption in which each TF bin is dominated by (at most) a single speaker.The training is based on simulated data generated by a publicly available RIR generator software 1 , efficiently implementing the image method [26].A four microphone linear array was simulated with (8, 8, 8) cm inter-microphone distances.Similar microphone inter-distances were used in the test phase.For each training sample, the acoustic conditions were randomly drawn from one of the simulated rooms of different sizes and different reverberation levels RT 60 as described in Table 2.The microphone array was randomly placed in the room in one out of six arbitrary positions.
For each scenario, two clean signals were randomly drawn from the Wall Street Journal 1 (WSJ1) database [27] and then convolved with RIRs corresponding to two 1 Available online at github.com/ehabets/RIR-Generator.The contributions of the two sources were then summed with a random signal to interference ratio (SIR) selected in the range of SIR ∈ [−2, 2] to obtain the received microphone signals.Next, we calculated the STFT of both the mixture and the STFT of the separate signals with a framelength K = 512 and an overlap of 75% between two successive frames.
We then constructed the audio feature tensor R as described above.In the training phase, both the location and a clean recording of each speaker were known; hence, they could be used to generate the labels.For each TF bin (l, k), the dominant speaker was determined by:
The ground-truth label D l,k is the DOA of the dominant speaker.The training set comprised 4 h of recordings with 30,000 different scenarios of mixtures of two speakers.It is worth noting that as the length of each speaker recording was different, the utterances may also include nonspeech or single-speaker frames.The network was trained to minimize the cross-entropy between the correct and the estimated DOA.The cross-entropy cost function was summed over all the images in the training set.The network was implemented in Tensorflow with the ADAM Table 1 The TF-DOAnet multi-speaker localization algorithm
• Compute the iRTF features from the multi-microphone recordings.
• Apply the U-net network to classify each TF bin to one of the possible DOAs.
• Based on the U-net results, decide the locations of the active speakers at each time frame.
optimizer [28].The number of epochs was set to be 100, and the training stopped after the validation loss increased for 3 successive epochs.The mini-batch size was set to be 64 images.
3 Experimental study

## Datasets

We evaluated the TF-DOAnet and compared its performance to both classic and DNN-based algorithms.To objectively evaluate the performance of the TF-DOAnet, we first simulated two rooms that were different from the rooms in the training set.Then, we tested our TF-DOAnet with real RIR recordings in different rooms.Finally, a reallife scenario with fast moving speakers was recorded and tested.For each test scenario, we selected two speakers from the test set of the WSJ1 database [27] and placed them at two different angles between 0 and 180 • relative to the microphone array, at a distance of either 1 m or 2 m.The signals were generated by convolving the signals with RIRs corresponding to the source positions and with either simulated or recorded acoustic scenarios.The SIR was tested in accordance with the DOA literature.

## Performance measures

Two different measures to objectively evaluate the results were used: the mean absolute error (MAE) and the localization accuracy (Acc.).The MAE, computed between the true and estimated DOAs for each evaluated acoustic condition, is given by
where N is the number of simultaneously active speakers and C is the total number of speech mixture segments considered for evaluation for a specific acoustic condition.The term π is the permutation and S N represents the permutation possibilities.The true and estimated DOAs for the nth speaker in the cth mixture are denoted by θ c n and θc n , respectively.The localization accuracy is given by Acc.(%) = Ĉacc.

## C

× 100 (10) where Ĉacc.denotes the number of speech mixtures for which the localization of the speakers is accurate.We considered the localization of speakers for a speech frame to be accurate if the angular distance between the true and the estimated DOA for all the speakers was less than or equal to 5 • .

## Compared algorithms

We compared the performance of the TF-DOAnet with two frequently used baseline methods, namely the MUSIC and SRP-PHAT algorithms.In addition, we compared its performance with the CNN multi-speaker DOA (CMS-DOA) estimator [5] 2 .To facilitate the comparison, the MUSIC pseudo-spectrum was computed for each frequency subband and for each STFT time frame, with an angular resolution of 5, over the entire DOA domain.
Then, it was averaged over all frequency subbands to obtain a broadband pseudo-spectrum followed by averaging over all the time frames L. Next, the two DOAs with the highest values were selected as the final DOA estimates.Similar post-processing was applied to the computed SRP-PHAT pseudo-likelihood for each time frame.

## Speaker localization results

## Static simulated scenario

We first generated a test dataset with simulated RIRs.Two different rooms were used, as described in Table 3.For each scenario, two speakers (male or female) were randomly drawn from the WSJ1 test database and placed at two different DOAs within the range {0, 5, . . ., 180} relative to the microphone array.Since the length of each speaker recording is different, the test dataset also includes non-speech or single-speaker frames.We assume the minimum angle between 2 speakers to be 20 • , which, for the radius of ≈ 1.5 m from the microphone array, implies that the speakers are practically standing shoulder to shoulder.Each speaker has a different signal length in the mixture.The microphone array was similar to the one used in the training phase.The assumption that we are familiar with the microphone array is fairly common and realistic.For instance, the microphone array in a conference room, in smart devices, or even in phones, is known in advance.Using the RIR generator, we generated the RIR for the given scenario and convolved it with the speakers' signals.
The results for the TF-DOAnet compared with the competing methods are depicted in Table 4.The tables demonstrate that the deep-learning approaches outperform the classic approaches.The TF-DOAnet achieved very high scores and outperforms the DNN-based CMS-DOA algorithm in terms of both MAE and accuracy.Note that the results in Table 4 are reported at a frame-based resolution, where each frame may consist one or two speakers.

## Static real recordings scenario

The best way to evaluate the capabilities of the TF-DOAnet is testing it with real-life scenarios.For this purpose, we first carried out experiments with real measured RIRs from a multi-channel impulse response database [19], recorded in our lab.The database comprises RIRs measured in an acoustics lab for three different reverberation times of RT 60 = 0.160, 0.360, and 0.610 s.The lab dimensions are 6 × 6 × 2.4 m.
The recordings were carried out with different DOA positions in the range of [ 0 • , 180 • ], in steps of 15 • .The sources were positioned at distances of 1 m and 2 m from the center of the microphone array.The recordings were  The results for the TF-DOAnet compared with the competing methods are depicted in Table 5.Again, the TF-DOAnet outperforms all competing methods, including the CMS-DOA algorithm.Note that the results are reported per time frame and not per utterance, and hence, the inferior results may be expected.Interestingly, for the 1 m case, the best results for the TF-DOAnet were obtained for the highest reverberation level, namely RT 60 = 610 ms, and for the 2 m case, for RT 60 = 360 ms.While surprising at the first glance, this can be explained using the following arguments.There is an accumulated evidence that reverberation, if properly addressed, can be beneficial in speech processing, specifically for multi-microphone speech enhancement and source extraction [20,29,30] and for speaker localization [31,32].In reverberant environments, the intricate acoustic propagation pattern constitutes a specific "fingerprint" characterizing the location of the speaker(s).When reverberation level increases, this fingerprint becomes more pronounced and is actually more informative than its an-echoic counterpart.An inference methodology that is capable of extracting the essential driving parameters of the RIR will therefore improve when the reverberation is higher.If the acoustic propagation becomes even more complex, as is the case of high reverberation and a remote speaker, a slight performance degradation may occur, but as evident from the localization results, for sources located 2 m from the array, the performance for RT 60 = 610 ms is still better than the performance for RT 60 = 160 ms.
It is worth noting that the test samples were not part of the training phase.The network was not fine-tuned for these test conditions.Yet, since we trained the network with the same RIR generator (with different conditions), it is likely that the results on the simulated test set will be high.The RIR generator cannot capture the accurate sound propagation in real acoustic environments.Therefore, with real recordings, the network performance is likely to be inferior.

## Real-life dynamic scenario

To further assess the capabilities of the TF-DOAnet, we also carried out experiments in real dynamic scenarios.The recordings took place at the acoustic lab, Bar-Ilan University, for which the reverberation level can be set in a wide range.We examined two reverberation levels, namely RT 60 = 390 ms and RT 60 = 720 ms.The microphone array consisted of 4 microphones with an inter-microphone spacing of 8 cm.The speakers walked naturally on an arc at a distance of about 2.2 m from the center of the microphone array.For each RT 60 , two experiments were recorded.The two speakers started at the angles 20 • and 160 • and walked until they reached 70 • and 100 • , respectively, turned back and walked to their starting point.This was done several times throughout the recording.The input SIR values of the first and second speakers are SIR = −0.12,0.12 dB, respectively; hence, both speakers have almost identical power.In the first room setup (RT 60 = 390 ms), the speed of the two moving speakers was 0.34 and 0.35 m/s, respectively.For the second setup (RT 60 = 720 ms), the speakers' speed was 0.28 and 0.31 m/s, respectively.Figure 2a depicts the real-life experiment setup and Fig. 2b depicts a schematic diagram of the setup of this experiment.The ground truth labels  of this experiment were measured with the Marvelmind indoor 3D tracking set 3 .Figures 3 and 4 depict the results of the two experiments.It is clear that the TF-DOAnet outperformed the CMS-DOA algorithm, especially for the high RT 60 conditions.Whereas the CMS-DOA fluctuated rapidly, the TF-DOAnet output trajectory was smooth and noiseless.
Table 6 depicts the computational cost of the proposed algorithm in comparison to the CMS-DOA algorithm.It is evident that the number of parameters used the network of the proposed model is less than half of the respective number of parameters of the CMS-DOA model.Moreover, the processing time of the proposed method is also slightly shorter.Note that the processing of 1-s-long utterance takes 70 ms on NVIDIA DGX V100 (single GPU) machine.

## Blind source separation of dynamical speakers

We next evaluate the applicability of the proposed method to the challenging task of speaker separation.Single microphone approaches, as they only utilize spectral information, have the potential of being robust to the source movement.However, their performance is rapidly deteriorating in reverberant environments [33].Multichannel speaker separation algorithms can remarkably separate overlapping speakers in static scenarios [34].In dynamic scenarios, the acoustic propagation from the sources to the microphones are rapidly changing over time.Tracking these acoustic paths is a cumbersome task, and failing to do so may result in significant performance degradation.
We propose here a new blind source separation approach, which can be implemented as a byproduct of the proposed tracking scheme.First, the estimated number of speakers, N, is inferred by selecting directions θ for which p l (θ) > 0.15 (see Eq. 7).For each speaker, the tracking path, θi (l), is found as explained in the previous section.TF masks, Mi (l, k), i = 1, . . ., N are obtained for each tracking path, as explained below.
We first aggregate probabilities from adjacent DOAs:
Then, we apply a threshold to this mask to mitigate the musical noise phenomenon:
3 https://marvelmind.com/product/starter-set-ia-02-3d/To circumvent source permutation issues, we maintain track smoothness by associating DOA estimates with a specific source, only if the current estimate is within 10 • of the estimate at the previous frame.Other, more sophisticated, tracking schemes can be applied, but the heuristic approach proposed here provided satisfactory results for the examined scenarios.More involved scenarios, such as intersecting trajectories that necessitate sophisticated tracking schemes, e.g., Bayesian methods [35,36], are left for a future study.
Once the TF masks are obtained, the separation is implemented by applying the masks to z ref , the mixed signal in the reference microphone.
Figure 5 depicts the mixed signal, described at the previous section, the estimated TF masks and the separated signals.To estimate the masks, we used the tracking path from Fig. 3c.The separation capabilities are clearly demonstrated from these figures.After the application of the proposed algorithm, the output SIR values of the first and second speakers are, respectively, SIR = 6.08 dB and SIR = 7.51 dB, i.e., approximately 7 dB improvement.It is worth noting that separating overlapping dynamic speakers in a highly reverberant room is a challenging task and the obtained results are promising.The reader is also referred to the corresponding audio samples in our website 4 .
A note on the validity of the WDO assumption [22] is in place.This widely used assumption underlies many blind audio separation algorithms that apply binary masking.Strictly speaking, this assumption may not hold in reverberant environments for multiple time-frequency bins, due to the "smearing" effect of the reverberation phenomenon.While this may only marginally degrade localization performance in static environments, it can significantly deteriorate speaker separation capabilities, especially in dynamic scenarios.In our experiments, we have shown that even a naïve application of timefrequency masking (see Eq. ( 12)) can yield satisfactory separation performance.Other, more sophisticated separation schemes that utilize these masks may be applied.Such schemes are left for a future study.

## Conclusions

A joint time-frequency approach was presented in this paper for the DOA estimation task.Instantaneous RTF features were used to train the model.The high TF resolution facilitated the simultaneous tracking of multiple moving speakers.A comprehensive experimental study was carried out with both simulated and real-life recordings.The proposed approach outperformed both the classic and CNN-based SOTA algorithms in all experiments.As a byproduct of the DOA tracking algorithm, we also presented a separation scheme, based on TF masking, which can be applied to moving speakers in a reverberant environment.We believe that the proposed method can be also applicable for localization audio signals other than speech [37].

## Fig. 1

Fig. 1 Block diagram of the TF-DOAnet algorithm.The dashed envelope describes the feature extraction step

## Fig. 2 Fig. 3 Fig. 4

Fig. 2 Real-life experiment setup

## Fig. 5

Fig. 5 Real-life separation results of two moving speakers in a 6 × 6 × 2.4 room with RT 60 = 390 ms

## Abbreviations

AbbreviationsTF-DOAnet: Time-frequency direction-of-arrival net; WDO: W-disjoint orthogonality; SOTA: State-of-the-art; FCN: Fully convolutional network; DDESS: Deep direction estimation for speech separation ; NMF: Non-negative matrix factorization; SDR: Signal to distortion ratio; TF: Time-frequency; BF: Beamformer; BSS: Blind source separation; DOA: Direction of arrival; SPI: Speaker position identifier; GSC: General sidelobe canceller; DSE: Deep single expert; MVDR: Minimum variance distortionless response; GEVD: Generalized eigenvalue decomposition; AIR: Acoustic impulse response; PSD: Power spectral density; cPSD: Cross-power spectral density; FIR: Finite-impulse response; MTF: Multiplicative transfer function; RIR: Room impulse response; LTI: Linear time invariant; DNN: Deep neural network; CNN: Convolutional neural network; MFCC: Mel-frequency cepstral coefficients; MMSE: Minimum mean square error; ASR: Automatic speech recognition; ATF: Acoustic transfer function; LCMV: Linearly constrained minimum variance; RTF: Relative transfer function; iRTF: Instantaneous relative transfer function; VAD: Voice activity detector; STSA: Short-time spectral amplitude estimator; LSAE: Log-spectral amplitude estimator; OMLSA: Optimally modified log spectral amplitude; IMCRA: Improved minima controlled recursive averaging; STFT: Short-time Fourier transform; DFT: Discrete Fourier transform; MoG: Mixture of Gaussians; MOE: Mixture of experts; MODE: Mixture of deep experts; r.v.: Random variable;

## Table 2

Configuration of training data generation.All rooms are 2.7 m in height

## Table 3

Configuration of test data generation.All rooms are 3 m in height

## Table 4

Results for two different test rooms with simulated RIRs

## Table 5

Results for three different rooms at distances of 1 m and 2 m with measured RIRs

## Table 6

Computational cost comparison

## Figures (text descriptions)

### Figure 1

Fig. 1 Block diagram of the TF-DOAnet algorithm.The dashed envelope describes the feature extraction step

Fig. 1
1
Fig. 1 Block diagram of the TF-DOAnet algorithm.The dashed envelope describes the feature extraction step

### Figure 2

Fig. 2 Real-life experiment setup

Fig. 2 Fig. 3 Fig. 4
234
Fig. 2 Real-life experiment setup

### Figure 3

Fig. 5 Real-life separation results of two moving speakers in a 6 × 6 × 2.4 room with RT 60 = 390 ms

Fig. 5
5
Fig. 5 Real-life separation results of two moving speakers in a 6 × 6 × 2.4 room with RT 60 = 390 ms

### Figure 4

AbbreviationsTF-DOAnet: Time-frequency direction-of-arrival net; WDO: W-disjoint orthogonality; SOTA: State-of-the-art; FCN: Fully convolutional network; DDESS: Deep direction estimation for speech separation ; NMF: Non-negative matrix factorization; SDR: Signal to distortion ratio; TF: Time-frequency; BF: Beamformer; BSS: Blind source separation; DOA: Direction of arrival; SPI: Speaker position identifier; GSC: General sidelobe canceller; DSE: Deep single expert; MVDR: Minimum variance distortionless response; GEVD: Generalized eigenvalue decomposition; AIR: Acoustic impulse response; PSD: Power spectral density; cPSD: Cross-power spectral density; FIR: Finite-impulse response; MTF: Multiplicative transfer function; RIR: Room impulse response; LTI: Linear time invariant; DNN: Deep neural network; CNN: Convolutional neural network; MFCC: Mel-frequency cepstral coefficients; MMSE: Minimum mean square error; ASR: Automatic speech recognition; ATF: Acoustic transfer function; LCMV: Linearly constrained minimum variance; RTF: Relative transfer function; iRTF: Instantaneous relative transfer function; VAD: Voice activity detector; STSA: Short-time spectral amplitude estimator; LSAE: Log-spectral amplitude estimator; OMLSA: Optimally modified log spectral amplitude; IMCRA: Improved minima controlled recursive averaging; STFT: Short-time Fourier transform; DFT: Discrete Fourier transform; MoG: Mixture of Gaussians; MOE: Mixture of experts; MODE: Mixture of deep experts; r.v.: Random variable;

Abbreviations

AbbreviationsTF-DOAnet: Time-frequency direction-of-arrival net; WDO: W-disjoint orthogonality; SOTA: State-of-the-art; FCN: Fully convolutional network; DDESS: Deep direction estimation for speech separation ; NMF: Non-negative matrix factorization; SDR: Signal to distortion ratio; TF: Time-frequency; BF: Beamformer; BSS: Blind source separation; DOA: Direction of arrival; SPI: Speaker position identifier; GSC: General sidelobe canceller; DSE: Deep single expert; MVDR: Minimum variance distortionless response; GEVD: Generalized eigenvalue decomposition; AIR: Acoustic impulse response; PSD: Power spectral density; cPSD: Cross-power spectral density; FIR: Finite-impulse response; MTF: Multiplicative transfer function; RIR: Room impulse response; LTI: Linear time invariant; DNN: Deep neural network; CNN: Convolutional neural network; MFCC: Mel-frequency cepstral coefficients; MMSE: Minimum mean square error; ASR: Automatic speech recognition; ATF: Acoustic transfer function; LCMV: Linearly constrained minimum variance; RTF: Relative transfer function; iRTF: Instantaneous relative transfer function; VAD: Voice activity detector; STSA: Short-time spectral amplitude estimator; LSAE: Log-spectral amplitude estimator; OMLSA: Optimally modified log spectral amplitude; IMCRA: Improved minima controlled recursive averaging; STFT: Short-time Fourier transform; DFT: Discrete Fourier transform; MoG: Mixture of Gaussians; MOE: Mixture of experts; MODE: Mixture of deep experts; r.v.: Random variable;

### Figure 5

Configuration of training data generation.All rooms are 2.7 m in height

Table 2
2
Configuration of training data generation.All rooms are 2.7 m in height
Simulated training dataRoom 1Room 2Room 3Room 4Room 5Room size(6 × 6) m(5 × 4) m(10 × 6) m(8 × 3) m(8 × 5) mRT 600.3 s0.2 s0.8 s0.4 s0.6 sSignalNoiseless signals from WSJ1 training databaseArray position in room6 arbitrary positions in each roomSource-array distance1.5 m with added noise with 0.1 variance

### Figure 6

Configuration of test data generation.All rooms are 3 m in height

Table 3
3
Configuration of test data generation.All rooms are 3 m in height
Simulated test dataRoom 1Room 2Room size(5 × 7) m(9 × 4) mRT 600.38 s0.7 sSource-array distance1.3 m1.7 mSignalNoiseless signals from WSJ1 test databaseArray position in room

### Figure 7

Results for two different test rooms with simulated RIRs

Table 4
4
Results for two different test rooms with simulated RIRs
Test roomRoom 1Room 2MeasureMAEAcc.MAEAcc.MUSIC [2]27.9528.3431.6218.38SRP-PHAT [3]28.2527.2336.6136.28CMS-DOA [5]12.8773.0924.039.25TF-DOAnet (our algorithm)1.5897.452.7693.0carried out with a linear microphone array consisting of8 microphones with three different microphone spacings.For our experiment, we chose the [8, 8, 8, 8, 8, 8, 8] cmsetup. In order to construct an array setup identical to theone in the training phase, we selected a sub-array of thefour center microphones out of the total 8 microphones inthe original setup. Consequently, we used a uniform lineararray (ULA) M = 4 elements with an inter-microphonedistance of 8 cm.

### Figure 8

Results for three different rooms at distances of 1 m and 2 m with measured RIRs

Table 5
5
Results for three different rooms at distances of 1 m and 2 m with measured RIRs
Distance1 m

### Figure 9

Computational cost comparison

Table 6
6
Computational cost comparison
# of parametersAverage inference time[million]of 1-s signal [seconds]CMS-DOA8.70.09TF-DOAnet2.10.07

## References

1. Multiple emitter location and signal parameter estimation. R Schmidt, IEEE Trans. Antennas Propag. 3431986

2. Broadband MUSIC: opportunities and challenges for multiple source localization. J P Dmochowski, J Benesty, S Affes, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics. 2007

3. A robust method for speech signal time-delay estimation in reverberant rooms. M S Brandstein, H F Silverman, IEEE International Conference on Acoustics, Speech and Signal Processing. 19971

4. The LOCATA challenge: acoustic source localization and tracking. C Evers, H W Löllmann, H Mellmann, A Schmidt, H Barfuss, P A Naylor, W Kellermann, IEEE/ACM Trans. Audio Speech Lang. Process. 282020

5. Multi-speaker DOA estimation using deep convolutional networks trained with noise signals. S Chakrabarty, E A P Habets, IEEE J. Sel. Top. Signal Process. 1312019

6. A learning-based approach to direction of arrival estimation in noisy and reverberant environments. X Xiao, S Zhao, X Zhong, D L Jones, E S Chng, H Li, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2015

7. A neural network based algorithm for speaker localization in a multi-room environment. F Vesperini, P Vecchiotti, E Principi, S Squartini, F Piazza, IEEE International Workshop on Machine Learning for Signal Processing (MLSP). 2016

8. Discriminative multiple sound source localization based on deep neural networks using independent location model. R Takeda, K Komatani, IEEE Spoken Language Technology Workshop (SLT). 2016

9. Source localization in reverberant rooms using deep learning and microphone arrays. H Pujol, E Bavu, A Garcia, International Congress on Acoustics (ICA). 2019

10. Towards end-to-end acoustic localization using deep learning: from audio signals to source position coordinates. J M Vera-Diaz, D Pizarro, J Macias-Guarasa, Sensors. 181034182018

11. Broadband DOA estimation using convolutional neural networks trained with noise signals. S Chakrabarty, E A P Habets, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). 2017

12. On the approximate W-disjoint orthogonality of speech. S Rickard, O Yilmaz, IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2002

13. O Ronneberger, P Fischer, T Brox, International Conference on Medical Image Computing and Computer-Assisted Intervention. U-net: convolutional networks for biomedical image segmentation. Springer2015

14. Speech dereverberation using fully convolutional networks. O Ernst, S E Chazan, S Gannot, J Goldberger, The 26th European Signal Processing Conference (EUSIPCO). 2018

15. Multi-microphone speaker separation based on deep DOA estimation. S E Chazan, H Hammer, G Hazan, J Goldberger, S Gannot, European Signal Processing Conference (EUSIPCO). 2019

16. Comparative analysis of the usage of neural networks for sound processing. S D Grechkov, V P Semenov, A A Bezrukov, IEEE Conference of Russian Young Researchers in Electrical and Electronic Engineering (EIConRus). 2020

17. Y Zhang, Q Duan, Y Liao, J Liu, R Wu, B Xie, The 4th International Conference on Mechanical. 2019

18. Attention wave-U-net for speech enhancement. R Giri, U Isik, A Krishnaswamy, IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). 2019

19. Multichannel audio database in various acoustic environments. E Hadad, F Heese, P Vary, S Gannot, International Workshop on Acoustic Signal Enhancement (IWAENC). 2014

20. Signal enhancement using beamforming and nonstationarity with applications to speech. S Gannot, D Burshtein, E Weinstein, IEEE Trans. Signal Process. 4982001

21. A minimum variance beamformer for spatially distributed microphones using a soft reference selection. S Stenzel, J Freudenberger, G Schmidt, 4th Joint Workshop on Hands-free Speech Communication and Microphone Arrays (HSCMA). 2014

22. Blind separation of speech mixtures via time-frequency masking. O Yilmaz, S Rickard, IEEE Trans. Signal Process. 5272004

23. Segnet: a deep convolutional encoder-decoder architecture for image segmentation. V Badrinarayanan, A Kendall, R Cipolla, IEEE Trans. Pattern Anal. Mach. Intell. 39122017

24. O Ronneberger, P Fischer, T Brox, International Conference on Medical Image Computing and Computer-assisted Intervention. U-net: convolutional networks for biomedical image segmentation. 2015

25. Supervised speech separation based on deep learning: an overview. D Wang, J Chen, IEEE/ACM Trans. Audio Speech Lang. Process. 26102018

26. Image method for efficiently simulating small-room acoustics. J B Allen, D A Berkley, J. Acoust. Soc. Am. 6541979

27. D B Paul, J M Baker, Workshop on Speech and Natural Language. The design for the Wall Street Journal-based CSR corpus. 1992

28. D P Kingma, J Ba, arXiv:1412.6980Adam: a method for stochastic optimization. 2014arXiv preprint (DOI: arXiv:1412.6980)

29. Multichannel eigenspace beamforming in a reverberant noisy environment with multiple interfering speech signals. S Markovich-Golan, S Gannot, I Cohen, 10.1109/TASL.2009.2016395IEEE Trans. Audio Speech Lang. Process. 1762009 (DOI: 10.1109/TASL.2009.2016395)

30. Raking the cocktail party. I Dokmanić, R Scheibler, M Vetterli, IEEE J Sel. Top. Signal Process. 952015

31. Acoustic space learning for sound-source separation and localization on binaural manifolds. A Deleforge, F Forbes, R Horaud, Int. J. Neural Syst. 250114400032015

32. Semi-supervised sound source localization based on manifold regularization. B Laufer-Goldshtein, R Talmon, S Gannot, IEEE/ACM Trans. Audio Speech Lang. Process. 2482016

33. Single channel voice separation for unknown number of speakers under reverberant and noisy settings. S E Chazan, L Wolf, E Nachmani, Y Adi, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). 2021

34. A consolidated perspective on multimicrophone speech enhancement and source separation. S Gannot, E Vincent, S Markovich-Golan, A Ozerov, IEEE/ACM Trans. Audio Speech Lang. Process. 2542017Invited tutorial paper

35. Microphone array speaker localizers using spatial-temporal information. S Gannot, T G Dvorkind, EURASIP J. Adv. Signal Process. 2006. 2006

36. A hybrid approach for speaker tracking based on TDOA and data-driven models. B Laufer-Goldshtein, R Talmon, S Gannot, IEEE/ACM Trans. Audio Speech Lang. Process. 2642018

37. A new non-linear framework for localization of acoustic sources. A K Das, T T Lai, C W Chan, C K Y Leung, Struct. Health Monit. 1822019
