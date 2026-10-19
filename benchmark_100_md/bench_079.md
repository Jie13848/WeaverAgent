# A BRIEF HISTORY OF AI: HOW TO PREVENT ANOTHER WINTER (A CRITICAL REVIEW)

```meta
arxiv: 2109.01517
```

## Authors

- Amirhosein Toosi
- Department of Integrative Oncology BC Cancer Research Institute Vancouver, BC, V5Z 1L3 atoosi@bccrc.ca
- Andrea Bottino
- Department of Computer and Control Eng. Polytechnic University of Turin Turin, Italy 10129 andrea.bottino@polito.it
- Babak Saboury
- Department of Radiology and Imaging Sciences National Institutes of Health Bethesda, MD 20892 babak.saboury@nih.gov
- Eliot Siegel
- Department of Diagnostic Radiology and Nuclear Medicine University of Maryland School of Medicine Baltimore, MD 21201 esiegel@umaryland.edu
- Arman Rahmim
- Departments of Radiology and Physics University of British Columbia Vancouver, BC arman.rahmim@ubc.ca
- December 12, 2022

## Abstract

The field of artificial intelligence (AI), regarded as one of the most enigmatic areas of science, has witnessed exponential growth in the past decade including a remarkably wide array of applications, having already impacted our everyday lives. Advances in computing power and the design of sophisticated AI algorithms have enabled computers to outperform humans in a variety of tasks, especially in the areas of computer vision and speech recognition. Yet, AI’s path has never been smooth, having essentially fallen apart twice in its lifetime (‘winters’ of AI), both after periods of popular success (‘summers’ of AI). We provide a brief rundown of AI’s evolution over the course of decades, highlighting its crucial moments and major turning points from inception to the present. In doing so, we attempt to learn, anticipate the future, and discuss what steps may be taken to prevent another ‘winter’. Keywords  Artificial intelligence · machine learning · deep learning · artificial neural networks · perceptron

## 1 Introduction

Artificial Intelligence (AI) technology is sweeping the globe, leading to bold statements by notable figures: “[AI] is going to change the world more than anything in the history of mankind” [1], “it is more profound than even electricity or fire” [2], and “just as electricity transformed almost everything 100 years ago, today I actually have a hard time thinking of an industry that I don’t think AI will transform in the next several years” [3]. Every few weeks there is news about AI breakthroughs. Deep-fake videos are becoming harder and harder to tell apart from real ones [4] [5]. Intelligent algorithms are beating humans in a greater variety of games more easily. For the first time in history, in arguably the most complex board game (named “Go”), DeepMind’s AlphaGo has beaten the world champion. AI has been around for decades, enduring “hot and cold” seasons, and like any other field in science, AI developments indeed stand on the shoulders of giants (see figure 1). With these in mind, this article aims to provide a picture of what AI essentially is and the story behind this rapidly evolving and globally engaged technology. Figure 1: A summary of the influence of different fields on AI.

## 2 What is AI?

AI is “the theory and development of computer systems able to perform tasks normally requiring human intelligence, such as visual perception, speech recognition, decision-making, and translation between languages” [6]. Marvin Minsky, American mathematician, computer scientist, and famous practitioner of AI defines AI as “the science of making machines do things that would require intelligence if done by men” [7]. John McCarthy who coined the term “artificial intelligence” in 1956, described it as “the science and engineering of making intelligent machines”. IBM suggests that “Artificial intelligence enables computers and machines to mimic the perception, problem-solving, and decision-making capabilities of the human mind” [8]. McKinsey &amp; Company explains it as a “machine’s ability to mimic human cognitive functions, including perception, reasoning, learning, and problem-solving” [9]. Figure 2: Summary of two-dimensional AI approaches as proposed by Russel and Norvig [10] (Data from Norvig, Peter, and Russell, Stuart. Artificial Intelligence: A Modern Approach, EBook, Global Edition. United Kingdom, Pearson Education, 2021.) Russel and Norvig [11] proposed four conceivable approaches to AI: acting humanly, thinking humanly, acting rationally, and thinking rationally (see figure 2). The British mathematician Alan Turing published a paper in 1950 (“Computers and intelligence” [12]) in which he proposed a tool to determine the difference between a task performed by a person and a machine. This test, known as the “Turing test”, consists of a series of questions to be answered. A computer can pass the test if a human interrogator cannot tell whether the answers to the questions come from a person or a computer. As such, to pass the test, the computer is required to have a number of essential capabilities such as: Natural language processing - to manage a natural and effective communication with human beings; Knowledge representation - to store the information it receives; Automated reasoning - to perform question answering and update the conclusion; And machine learning - to adjust to new situations and recognize new patterns. In Turing’s view, a physical simulation of a human is totally irrelevant to demonstrate intelligence. Other researchers, however, have suggested a complete Turing test [13, 14, 15] that involves interaction with real-world objects and people. Hence, the machine should be equipped with two additional (and vital) capabilities to pass the “Extended” version of the Turing test [10]: Computer vision and speech recognition - to see and hear the environment; And robotics - to move around and interact with the environment.

## 3 History of AI

The field of AI has experienced extreme ascends and descends over the last seven decades. These recurring ridges of great promise and valleys of disappointment referred to as AI’s Summers and Winters, have divided the history of AI into three distinct cycles (see figure 3). These different cycles and seasons will be discussed in the following parts.

## 3.1 Prehistoric Events

When science fiction writer Isaac Asimov wrote his timeless book "I, Robot" in 1942, he likely did not imagine that this work, 80 years later, would become a primary source for defining the laws governing human-robot interactions in modern AI ethics. Although Asimov’s novels (figure 4) are often considered as the birthplace of the ideas of intelligent machines [16], McCulloch and Pitts’ article, “A Logical Calculus of the Ideas Immanent in Nervous Activity” published in 1943 [17], was the first step toward the implementation of AI [18, 19, 20, 21]. Figure 3: Dividing the history of AI into three recurring cycles (public excitements/time). (Adapted from T. Noguchi et al., “A practical use of expert system “AI-Q” focused on creating training data,” 2018 5th International Conference on Business and Industrial Research (ICBIR), 2018, pp. 73-76; With permission. (Figure 1 in original).) (a) (b) Figure 4: (a): Isaac Asimov, the well-known sci-fi writer. (b): I, Robot, Asimov's sci-fi book series [22]. Based on Alan Turing's "On Computable Numbers [23]," their model provided a way to abstractly describe brain functions, and demonstrated that simple elements connected in a neural network can have enormous computational power. The article received little attention until John von Neumann, Norbert Wiener, and others applied its concepts. "McCulloch - Pitts" neuron, was the first mathematical model of an artificial neural network. This model, inspired by the basic physiology and function of the brain's neurons, showed that essentially any computable function could be modeled as a connected network of such neurons [10]. Based on this work, six years later, Donald Hebb proposed a simple learning rule to tune the strength of the neuron connections [24]. His learning method, namely "Hebbian learning," [25] is considered as the inspiring model for neural networks learning. Building upon these works, one year later, in the summer of 1950, two Harvard undergrad students, Marvin Minsky and Dean Edmonds built the first analog neural net machine called SNARC [26]. SNARC stands for “stochastic neural-analog reinforcement calculator” and was based on a network of 40 interconnected artificial hardware neurons built using 3000 vacuum tubes and the remains of a B-24 bomber’s automatic pilot mechanism. SNARC was successfully applied to find the way out from a maze (See figure 5). Figure 5: One node of 40 nodes constructing the stochastic neural-analog reinforcement calculator (SNARC) [27] AI developed significantly from the studies of Alan Turing (figure 6) during his short life, considered in all respects as one of the fathers of AI. Although Turing owes much of his fame to the work he did at the Bletchley Park center to decode German communications during World War II, his remarkable work toward the theory of computation dates back to his article published when he was only 24 [23]. Turing demonstrated that his “universal computing machine” could perform any imaginable mathematical computation if it could be represented as an algorithm. John von Neumann stated that Turing’s article laid the groundwork for the central concept of modern computers. A few years later, in 1950, in his article entitled “Computing machinery and intelligence” [12], Turing raised the fundamental question of “Can a machine think?”. The imitation game or the Turing test evaluates the ability of a machine to “think”. In this test, a human was asked to distinguish between a machine’s written answers and those of a human (figure 6). A machine is considered as being intelligent if the human interrogator could not tell if the answer is given by a human or a machine [28]. Before the term AI was coined, many works were pursued that were later recognized as AI, including two checkers-playing games, developed almost at the same time by Arthur Samuel at IBM and Christopher Strachey at the University of Manchester in 1952.

## 3.2 The First Summer of AI

The term AI was coined around six years after Turing’s paper [12], in the summer of 1956, when John McCarthy, Marvin Minsky, Claude Shannon, and Nathaniel Rochester gathered common interest in automata theory, neural networks, and cognitive science (2-month workshop at Dartmouth College). There, the term ‘Artificial Intelligence’ was coined by McCarthy. McCarthy defined AI as “the science and engineering of making intelligent machines,” emphasizing the parallel growth between computers and AI. The conference is sometimes referred to as the “birthplace of AI” because it coordinated and energized the field [10], and this time is considered as the beginning of an era called “the first summer of AI”. One of the consequent results of the Dartmouth Conference was the work of Newell and Simon. They presented a mathematics-based system for proving symbolic logic theories, called the Logic Theorist (LT), along with a list processing language for writing them called IPL (Information Processing Language) [29]. Soon after the conference, their program was able to prove most of the theorems (38 out of 52 of them) in the second chapter of Whitehead &amp; Russell’s “Principia Mathematica”. In fact, the program was able to give a solution for one of the theorems that was shorter than the one in the text. Newell and Simon later released their General Problem Solver (GPS) which was designed to mimic the problem-solving protocols of the human brain [30]. General problem solver is counted as the first work in the “reasoning humanly” framework of AI. Figure 6: (a): Alan Turing. (b): Schematic of the Turing Test. Using reinforcement learning, Arthur Samuel’s 1956 checker player quickly learned to play at an intermediate level, better than its own developer [31]. Reinforcement learning is a type of AI algorithm where an AI agent learns how to interact with its surrounding environment to achieve its goal through a reward-based system. He demonstrated his checker player program on television, making a great impression [32]. His work is considered to be the first reinforcement learning-based AI program, and indeed the forerunner of later systems such as TD-GAMMON in 1992, one of the world’s best backgammon players [33], and AlphaGo in 2016, which shocked the world by defeating the human world champion of Go [34]. A turning point in AI, and specifically in neural networks, occurred in 1957 when the psychologist researcher Frank Rosenblatt (considered a father of deep learning [35]) built the Mark I Perceptron at Cornell [36]. He built an analog neural network with the ability to learn through trial and error. More precisely, the perceptron was a single-layer neural network being able to classify the input data into two potential categories. The neural network produces a prediction, say “left” or “right”, and if it is incorrect, it attempts to get more accurate the following time. Accuracy increases with each iteration. A 5-ton IBM 704 computer the size of a room was fed by a large stack of punch cards (figure 7). The computer learned to identify cards on the left and cards on the right in 50 attempts. Mark I Perceptron is considered as one of the forerunners of modern neural networks [37]. In 1958 John McCarthy introduced the AI-specific programming language named LISP which became the prevailing AI programming language for the next three decades. In his article entitled “Programs with Common Sense”, he proposed a conceptual approach for AI systems based on knowledge representation and reasoning [39]. LISP is the first high-level AI programming language. In addition, 1958 is an important year because the first experimental work involving evolutionary algorithms in AI [40] was conducted by Freidberg toward automatic programming [41]. Nathaniel Rochester and Herbert Gelernter of IBM developed a geometry-theorem-proving program in 1959. Their AI-based program called “the geometry machine” was able to provide proofs for geometry theorems, which many math students had found quite tricky [42]. Written in FORTRAN, their “geometry machine” program is regarded as one of the first AI programs that could perform a task as well as a human. Another important event during the early 1960s is the emergence of the first industrial robot. Named “Unimate”, the robotic arm was used on an assembly line in General Motors in 1961 for welding and other metalworks [43]. In 1962, Widrow and Frank Rosenblatt revisited Hebb’s learning method. Widrow enhanced the Hebbian learning method in his network called Adaline [44] and Rosenblatt in his well-known Perceptrons [36]. Marvin Minsky in 1963 proposed a simplification approach for AI use cases [45]. Marvin Minsky and Seymour Papert suggested that AI studies concentrate on designing programs capable of intelligent behavior in smaller artificial environments. The so-called blocks universe, which consists of colored blocks of different shapes and sizes arranged on a flat surface, has been the subject of many studies. The framework called Microworld became a backbone for subsequent works. Instances are James Slagle’s SAINT program for solving Figure 7: Mark I Perceptron at the Smithsonian museum [38]. closed-form calculus integration problems in 1963 [46], Tom Evans’s ANALOGY in 1964 for solving an IQ test’s geometric problems [47], and the STUDENT program, written in LISP by Daniel Bobrow in 1967, for solving algebra problems [48]. ELIZA, the first chatbot in the history of AI was developed by Joseph Weizenbaum at MIT in 1966. ELIZA was designed to serve as a virtual therapist to ask questions and provide follow-ups in response to the patient [49]. SHAKEY, the first omni-purpose mobile platform robot was also developed at Stanford Research Institute in 1966 with reasoning about its surrounding environment [50].

## 3.3 The First Winter of AI

The hype and high expectations caused by the media and the public from one side, and the false predictions and exaggerations by the experts in the field about their outcome from the other side, led to major funding cuts in AI research in the late 1960s. Governmental organizations like Defense Advanced Research Projects Agency (DARPA) had already granted enormous funds for AI research projects during the 1960s. Two reports brought about major halts in supporting the research: the US government report, namely ALPAC in 1966 [51], and the Lighthill report of the British government in 1973 [52]. These reports mainly targeted the research pursued in AI, mostly the research works done on artificial neural networks and came up with a grim prediction for the technology’s prospects. As a result, both the US and the UK governments started to decrease support for AI research at universities. DARPA, which had previously funded various research projects in the 1960s, now needed specific timelines and concise explanations of each proposal’s deliverables. These events slowed the advancement of AI and ushered in the first AI winter, which lasted until the 1980s. It is crucial to recognize the three key factors that caused this major halt in AI research for that era. First, many early AI systems pursued the “thinking humanly” approach to solve the problems. In other words, instead of taking a bottom-up approach starting from thoroughly analyzing the task, providing a possible solution, and turning it into an implementable algorithm, they took the opposite direction, merely relying on replicating the way humans perform the task. Second, there was a failure to recognize the complexity of many of the problems. Resulting from the oversimplification of the AI frameworks proposed by Marvin Minsky, most early problem solving systems succeeded mainly on toy (simplistic) problems, by combining simple steps to come up with a solution. However, many of the real-world problems that AI was attempting to solve were in fact intractable. It was commonly assumed that “scaling up” to bigger problems was merely a matter of faster hardware and greater memory capacity. However, developments in computational complexity theory proved it wrong. The third factor was related to the negative conceptions about neural networks and the limitations of their fundamental structures. In 1969 Minsky pointed out the limited representational abilities of a perceptron, (to be exact, a single-layer perceptron cannot implement the classic XOR logical function), and despite not being a general critique about neural networks, this also contributed to global funding cuts in neural networks research.

## 3.4 Expert Systems; the Revival of AI, and the Second Summer

Mainstream AI research efforts during the previous two decades were generally based on so-called “weak AI”, that is, providing general solutions based on search algorithms in a space of all possible states built on basic reasoning steps. Despite being general purpose, these approaches suffered from a lack of scalability to larger or more complex domains. To address these drawbacks, in the early 1980s, researchers decided to take a more robust approach utilizing domain-specific information for stronger reasoning but in narrower areas of expertise. The new approach, so-called “expert systems”, originated at Carnegie Mellon University, and was quickly able to find its way to corporations. DENDRAL [53], created at Stanford by Ed Feigenbaum, Bruce Buchanan, and Joshua Lederberg in the late 1960s and early 1970s, and inferred molecular structure from mass spectrometry data, was an early success story. DENDRAL was the first effective knowledge-intensive system, relying on a vast range of special-purpose laws, to provide expertise rather than basic knowledge. In 1971 at Stanford University, Feigenbaum started the Heuristic Programming Project aimed at extending the area in which expert systems could be applied. The MYCIN system was one of the successful consequent results of the new wave, developed in the mid 1970s for the purpose of blood infection diagnosis by Edward Shortliffe under the supervision of Bruce Buchanan and Stanley Cohen. MYCIN could perform identification of bacteria causing sepsis, and recommend antibiotics dosage based on patient weight. It could perform diagnosis on par with the human experts in the field, and significantly better than medical interns, benefiting from around 600 deduced rules in the form of a knowledge base, from extensive interviews with the experts, by means of integrating uncertainty calculations [54]. Meanwhile, one of the most important moves toward deep convolutional neural networks (CNN) happened in 1980. The “neocogitron”, the first convolutional neural networks (CNN) architecture, was proposed by Fukushima in 1980 [55]. Several learning algorithms were suggested by Fukushima to train the parameters of a deep neocogitron so that it could learn internal representations of input data. This work is in fact regarded as the origin of today’s deep convolutional neural networks. R1, developed by McDermott in 1982, was the first successful commercial expert system used in the digital equipment industry, for the configuration of new computer systems’ orders [56]. In nearly 4 years, the firm added $40 million of revenue using R1. By 1988, most corporations in the United States benefited from expert systems, either by being a user of the system or doing research in the field [57]. The application of expert systems to real-world problems resulted in the development of a wide range of representations and reasoning tools. The Prolog language gained popularity in Europe and Japan, whereas the PLANNER language family thrived more in the United States. In Japan, the government started a 10-year plan to keep up with the new wave by investing more than $1.3 billion in intelligent systems. The US government, by establishing the Microelectronics and Computer Technology Corporation in 1982, revived AI research in hardware, chip design, and software research. The same change happened in the UK as well, resulting in reassignment of funds previously cut. All these events during the 80s led to a period of “Summer” for AI. The AI industry thrived from billions of dollars invested in the field, and various activities emerged from expert systems developer companies to domain-specific hardware, computer vision, and robotic systems. Overall, the AI industry boomed from a few million dollars in 1980 to billions of dollars in the late 1980s, including hundreds of companies building expert systems, vision systems, robots, and software and hardware specialized for these purposes.

## 3.5 The Second Winter of AI

Despite all efforts and investments made during the early 1980s, many companies could not fulfill their ambitious promises. Hardware manufacturers declined to keep up with the requirements of specialized needs of the expert systems. Hence, the thriving industry of expert systems in the early 1980s declined tremendously and inevitably collapsed by the end of the 1990s and the AI industry faced another winter that lasted until the mid 1990s. This second period of so-called “winter” in the history of AI had been so harsh that AI researchers subsequently tended to avoid even the term “AI” by choosing other titles such as “informatics” or “analytics”. Despite the big shutdown of AI-based research works, the second winter was the time when the very well-known backpropagation algorithm was revisited by many research groups [58] [59]. Backpropagation, which is a primary learning mechanism for artificial neural networks, was vastly used in learning problems during these years and eventually led to a new wave of interest in neural networks. The lesson learned during the periods of AI’s winter made researchers more conservative. As a result, during the late 1980s and the 1990s, the field of AI research witnessed a major conservative shift toward more established theories like statistics-based methods. Among these theories finding their way to the field were hidden Markov models (HMMs) [60]. Being strictly mathematical-based and resulting from extensive training on large real-world datasets, hidden Markov models became a trustable framework for AI research, especially in handwriting recognition and speech processing, helping them to make their way back to the industry. Another important outcome of this conservative shift in the field of AI was the development of public benchmark datasets and related competitions in its various subfields. Instances include the Letter Dataset [61], Yale face database [62], MNIST dataset [63], Spambase Dataset [64], ISOLET Dataset [65], TIMIT [66], JARTool experiment Dataset [67], Solar Flare Dataset [68], EEG Database [69], Breast Cancer Wisconsin (Diagnostic) Dataset [70], Lung Cancer Dataset [71], Liver Disorders Dataset [72], Thyroid Disease Dataset [73], Abalone Dataset [74], UCI Mushroom Dataset [75] and other datasets that have been gathered during the 1990s. The availability of these public benchmarks became an important means for rigorous measurement of AI research advancements.

## 3.6 Man versus Machine

The gradual public interest in AI during the early 90s opened doors to other emerging or established fields such as control theory, operational research, and statistics. Decision theory and probabilistic reasoning started being adopted by AI researchers. Uncertainty was represented more effectively by introducing Bayesian networks to the field [76]. Rich Sutton in 1998 revisited reinforcement learning after around thirty years by adopting Markov decision processes [77]. This step led to a growth in applying reinforcement learning on various problems, such as planning research, robotics, and process control. The vast amount of available data in different areas on the one hand, and the influences of statistical methods such as machine learning and optimization on AI research methods, on the other, resulted in significant readoption of AI in subfields including multiagent models, natural language processing, robotics, and computer vision. As such, new hopes for AI shaped again in the early 1990s. Eventually, in 1997, AI-equipped machines showed off their power against “Man” to the public [78]. Chess-playing AI software developed in IBM, called “Deep Blue”, eventually won over the great maestro chess world champion, Garry Kasparov. Broadcasted live, Deep Blue captured the public’s imagination once again toward AI systems of the future. The news was so breathtaking that IBM’s share values rose up to all-time highs [79].

## 3.7 Information Age: Enter Big Data

Massive advances in microchip manufacturing technologies in the late 1990’s led to emerging powerful computers, concurrent to the growth of the global Internet that generated massive amounts of data. This information included enormous unprocessed text, video, voice, and images, along with semiprocessed data such as geographical tracking, social media-related data, and electronic medical records, ushering in the era of big data [80]. In the computer vision area in 2009, the ImageNet dataset was created gathering millions of labeled images, significantly contributing to the field [81]. There was a new beginning of wide interests in AI from the industry. Notable steps were taken in 2011 when IBM’s Watson defeated human champions in the highly popular TV quiz show Jeopardy [82], significantly boosting public impression of the state-of-the-art in AI, and with the introduction of Apple’s Siri intelligent assistant.

## 3.8 Return of Neural Networks

In 1989 Yann LeCun revisited convolutional neural networks, and using gradient descend in their training mechanism, demonstrated the ability to perform well in computer vision problems, specifically in handwritten digit recognition [83]. Yet, it was in 2012 that these networks came to the forefront. A deep convolutional neural network developed in Geoffrey Hinton’s research group at the University of Toronto surpassed the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) competitors by significantly enhancing all ImageNet classification benchmarks [84]. Before the use of deep neural networks, all best-performing methods were mostly so-called classical machine learning methods using hand-crafted features. By 2011 the computing power of graphics processing units had grown enough to help the researchers train deep networks with higher dimensions both in terms of width and depth in a shorter time. Since the earlier implementation of Convolutional Neural Networks on graphics processing units in 2006 [85] which resulted in four times faster performance compared with central processing units, Schmidhuber’s team at IDSIA could achieve a 60 times faster performance on graphics processing units in 2011 [86]. Meanwhile, the availability of huge amounts of labeled data such as millions of labeled images in the ImageNet dataset helped researchers to overcome the problem of overfitting. Eventually, in 2012 Hinton’s team proposed a deep convolutional neural network architecture, named AlexNet (after the team’s leading author Alex Krizhevsky), which was able to train more layers of neurons. Using many mechanisms and techniques such as rectified linear unit activation functions and the dropout technique, AlexNet could achieve higher discriminative power in an end-to-end fashion, that is, to feed the network with merely the pure images of the dataset [87]. This event is regarded as the birth of the third boom in AI. Since then, deep learning-based methods have continued to achieve outstanding feats, including outperforming or performing on par with human experts in certain tasks. Instances include AI-related fields such as computer vision, natural language processing, medical image diagnosis [88], and natural language translation. The progress of deep neural networks gained public attention in 2016 at the time when Deep Mind’s AlphaGo beat the world champion of Go [34]. AI became again the target of the media, public, governments, industries, scholars, and investor’s interests. Deep learning methods have nowadays entirely dominated AI-related research, creating entirely new lines of research and industries. In 2018, Yoshua Bengio, Geoffrey Hinton, and Yann LeCun won the Turing award for their pioneering efforts in deep learning. Figure 8 summarizes the timeline of AI from the time it was born up to now. Figure 8: The timeline of developments in AI

## 4 Where Do We Stand Now?

The previous section presented a brief story of AI's journey, with all its ups and downs over the decades. This journey has not been easy, with multiple waves and seasons. Specifically, AI has faced two main breakdowns (so-called winters) and three main breakthroughs (so-called AI summers or booms). Thanks to the convergence of parallel processing, higher memory capacity, and more massive data collection (e.g., big data), AI has enjoyed a steady upward climb since the early 2020s. With all these pieces in place, much better algorithms have been developed, assisting this steady progression. Computers are becoming faster. Computing power has continued to double nearly every two years (Moore's law). Advancements in technology occur 10 times faster, that is, what used to take years, now may happen in the course of weeks or even days [89]. On a global scale, AI is becoming an attractive target for investors, producing billions of dollars of profit per annum. From 2010 to 2020, global investment in AI-based startup companies has steadily grown from $1.3 billion to more than $40 billion, with an average annual growth rate of nearly 50%, whereas in 2020 only, corporate investment in AI is reported to be nearly $70 billion globally [90]. In the academic sector, from 2000 up until 2020, the number of peer-reviewed AI articles per year has grown roughly 12 times worldwide. AI conferences have witnessed similar significant increases in terms of the number of attendants. In 2020, NeurIPS accepted 22,000 attendees, more than 40% growth over 2018, and 10-fold more compared with 2012. Concurrently, AI has become the most popular specialization among computer science PhD students in North America, nearly three times the next rival (theory and algorithms) [91]. In 2019, more than 22% of PhD candidates in computer science majored in AI and machine learning. With the introduction of machine learning, the environment of the health care and biology sectors has changed dramatically. AlphaFold, developed by DeepMind, used deep learning to make a major advance in the decades-long biology problem of protein folding. Scientists use machine learning algorithms to learn representations of chemical molecules to plan more efficient chemical synthesis. Machine learning-based approaches were used by PostEra, an AI startup, to speed up coronavirus disease 2019-related drug development during the pandemic [91]. This progress suggests that we are in the midst of the next hype cycle. And this new hype is focused on applications with life-or-death implications, such as autonomous vehicles, medical applications, and so on, making it critical that AI algorithms be trustworthy.

## 5 The Future of AI

Numerous AI-related startups have been founded in recent years, with both companies and governments investing heavily in the sector. If another AI winter occurs, many will lose their jobs, and many startups will be forced to close, as has occurred in the past. According to McKinsey &amp; Company, the economic gap between an approaching winter period and continued prosperity by 2025 would be in the tens of billions of dollars [92]. A recurrent pattern in previous AI winters has been the promises that sparked initial optimism yet turned out to be exaggerated. During both AI winters, budget cuts had a major effect on AI research. The Lighthill report resulted in funding cuts in the UK during the first AI winter, as well as cuts in Europe and the United States. DARPA support was cut, resulting in the second AI winter. Significant attention needs to be paid to technical challenges and limitations. Let us recall what was faced by the perceptron in the 1960s in being noted as unable to solve the so-called XOR problem, or limitations faced by expert systems in the 1980s. AI has appeared particularly vulnerable to overestimations coupled to technical limitations. Overall, the hype and fear that comes with reaching human-level intelligence have quickly contributed to exaggerations and public coverage that is not common in other innovative tech sectors. To avoid a next winter of AI, a number of important considerations may need to be made: i. It is extremely important to be aware of philosophical arguments about the utter sublimeness of what it means to be human, and to not make exaggerated claims about ascension of AI systems to being human (see very illustrating documentary [93]). In addition, these philosophical arguments (e.g. by Hubert Dreyfus based on the philosophy of Martin Heidegger), had they been more extensively and interactively considered, could have likely contributed to further success by AI in early years (e.g. earlier attention to ‘connectionist’ approaches to AI). 
 ii. Neglects regarding above point, as well as exciting early successes, contribute to exaggerated claims that AI will solve any important problem soon. This is what happened that contributed to the first winter of AI, where AI researchers made overconfident and overoptimistic predictions about upcoming successes, given the early promising performances of AI on simpler examples [94]. A lack of appreciation for the computational complexity theory was another reason for AI scientists to believe that scaling up of simple solutions to larger tasks is just a matter of using faster hardware and larger memories. 
 iii. According to a recently released report [95], 40% of startups established in Europe that claim to use AI in their provided services do not actually do so, largely because the definition of AI is ambiguous for the majority of the public and the media. Therefore, given recent excitement around AI and the resulting hype and investment growth in the field, some businesses try to benefit from this ambiguity by misusing terms such as AI, machine learning, and deep learning. Thus, it is crucial to define these terms more clearly with respect to other related concepts and define what they have in common and what they do not. 
 iv. There are significantly troubled trends in scientific methodology and dissemination by AI researchers, contributing to the hype and confusion: these trends include failure to distinguish between explanation and speculation, failure to identify real sources of performance gains, confusing/misleading use of math, and misuse of language [96]. According to a recent study [97] reviewing a spectrum of machine learning approaches for detecting and prognosticating coronavirus disease 2019 from standard-of-care chest radiographs and computed tomography images, none of the more than 400 studies were found suitable for clinical application! The studies suffered from one or multiple issues including use of poor-quality data, poor application of machine learning methodology, poor reproducibility and biases in study design. 
 v. AI, in its essence, is vague and covers a broad scope. As observed by Andrew Moore [98], “Artificial intelligence is the science and engineering of making computers behave in ways that, until recently, we thought only human intelligence is able to perform”. The critical point in this definition lies in the phrase “until recently” which points to the moving target of AI through time. In other words, ideas and methods are being referred to as AI as long as they have not been completely discovered. Once they are figured out, they may be no longer associated with AI and receive their own tag. This phenomenon is known as the AI effect [99] which contributes to the fast decline in public excitement about ground breaking achievements in AI. vi. An important challenge with AI technologies, more specifically deep learning-based AI, is its so-called black box, opaque nature of decision making. That is, when a deep learning algorithm makes a decision, the process of its inference or the logic behind it may not be representable. Although in some tasks such as playing board games, for example, “Go”, where the objective is merely winning the game, this concern does not reveal itself, in critical tasks; for example in health care, where a decision impacts humans lives, this issue can lead to trustworthiness challenges (other examples of critical tasks include transportation and mobility systems, and social or financial systems.) 
 vii. One very specific concern is issue of bias. As an example, soon after Google introduced bidirectional encoder representations from transformers (BERT), one of the most sophisticated AI technologies in language models, scientists learned an essential flaw in that system. BERT and its peers (GPT-2, GPT-3, t5, etc) were more likely to equate males with computer programming and, in general, failed to give females adequate respect. BERT, which is now being used in critical services such as Google’s Internet search engine, is one of a number of AI systems (so-called “transformers”) that learn from massive amounts of digitized data, such as old books, Wikipedia pages, tweets, forums, and news stories. The main problem with BERT or generative pretrained transformers (GPT-3), BERT’s rival introduced by OpenAI, and similar universal language models is that they are too complex even for their own designers (GPT-3’s full version has 175 billion parameters [100]); in fact, scientists are still learning how these models work. One certain fact about these systems is that they pick up biases as they learn through human-generated data. Because these mechanisms can be used in a variety of sensitive contexts to make critical and life-changing choices, it is critical to ensure that the decisions do not represent biased attitudes against particular communities or cultures. Thus, builders of AI systems have a duty to steer the design and use of AI in ways that serve society. Overall, the future of AI seems to be promising. AI eventually will drive our automobiles, aid physicians in making more precise diagnoses, assist judges in making more consistent judgments, help employers in hiring more qualified applicants, and much more. We are aware, however, that these AI systems may be fragile and unjust. By adding graffiti to a stop sign, the classifier may believe it is no longer a stop sign. By adding subtle noise or signal to a benign skin lesion image, the classifier may be fooled into believing it is malignant (eg, adversarial attacks). Risk management instruments used in US courts have been shown to be racially discriminatory. Corporate recruitment tools have been shown to be sexist. Toward trustworthy AI, organizations around the world are coming together to establish consistent standards for evaluating responsible implementation of AI systems and to encourage international support for AI technologies that benefit humanity and the environment. Among these tries is the European Commission’s report on Ethics guidelines for trustworthy AI [101] and DARPA’s XAI (eXplainable AI) roadmap [102]. Figure 9: Trade-off between interpretability and performance for AI models. (Adapted from Arrieta, Alejandro Barredo, et al. “Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI.”  Information Fusion  58 (2020): 82-115. With permission. (Figure 12 in original).) According to Arrieta and colleagues, [103], a trade-off between interpretability of AI models and their accuracy (performance) can be observed, given fair comparison conditions (figure 9). Simpler AI approaches, such as linear regression and decision trees, are self-explanatory (interpretable) because the classification decision border may be depicted in a few dimensions using model parameters. However, for tasks such as categorization of medical images in health care, these may lack the necessary complexity, yet to acquire the trust of physicians, regulators, and patients, a medical diagnostic system needs to be visible, intelligible, and explainable; it should be able to explain the logic of making a certain decision to stakeholders engaged in the process. Newer rules, such as the European General Data Protection Regulation, are making the use of black box models more difficult in different industries because retraceability of judgments is increasingly required. An AI system designed to assist professionals should be explainable and allow the human expert to retrace their steps and use their judgment. Some academics point out that humans are not always competent or willing to explain their choices. However, explainability is a fundamental enabler for AI deployment in the real world because it ensures that technology is used in a safe, ethical, fair, and trustworthy manner. Breaking AI misconceptions by demonstrating what a model primarily looked at while making a judgment can help end-users to trust the technology (eg, via use of heat maps/activation maps). For non-deep learning users, such as most medical professionals, it is even more vital to show such domain-specific attributes used in the decision. For further enhanced AI, a way forward seems to be the convergence of symbolic and connectionist methods, which would incorporate the former's higher interpretability with the latter's significant recent success (more on these next). For instance, the use of hybrid distributional models, which combine sparse graph-based representations with dense vector representations and connect those to lexical tools and knowledge bases, appears to be promising toward explainable AI in the medical domain [104]. However, the main obstacle toward this solution is the historical division between these two paradigms. The deep neural network approach is not novel. Today, it is fulfilling the promise stated at the beginning of cybernetics by benefitting from developments in computer processing and the existence of massive datasets. These techniques, however, have not always been deemed to constitute AI. Machine-learning approaches based on neural networks (connectionist AI) have been historically scorned and ostracized by the symbolic school of thinking. The rise of AI, which was clearly distinct from early cybernetics, amplified the friction between these two approaches. The cocitation network of the top-cited authors in articles mentioning AI demonstrates the drift between researchers who have used the symbolic or connectionist paradigms (See figure 10). Figure 10: Co-citation network of the 100 most cited authors with “Artificial Intelligence” in the title. Figure illustrates the names of some important authors, clearly distributed by the community. At the heart of the “connectionists,” some core figures in deep learning appear. On the “symbolic side,” some core figures are laid out in a way that represents their proximities and divergences, surrounded by primary contributors to the construction of cognitive modeling, expert systems, and even those critical of symbolic AI (Dreyfus, Searle, Brooks). This figure is not comprehensive and misses some key contributors; and is intended to demonstrate existing dichotomies in connectionist and symbolic frameworks.(From Cardon, Dominique, Jean-Philippe Cointet, and Antoine Mazières. “Neurons spike back. The invention of inductive machines and the artificial intelligence controversy”, Réseaux, vol. 211, no. 5, 2018, pp. 173-220. Available at:  https://neurovenge.antonomase.fr/ . Accessed May 18 2021; with permission.) Despite the obvious separation that existed among the intellectuals from these two schools, a third subfield of AI has been emerging, namely neuro-symbolic AI, which focuses on combining the neural and symbolic traditions in AI for additional benefit [105]. The promise of NeSy AI is largely based on the aim of achieving a best-of-both-worlds scenario in which the complementary strengths of neural and symbolic techniques can be advantageously merged. On the neural side, desirable strengths include trainability from raw data and robustness against errors in the underlying data, whereas on the symbolic side, one would like to retain these systems’ inherent high explainability and provable correctness, as well as the ease with which they can be designed and function using deep human expert knowledge. In terms of functional features, using symbolic approaches in conjunction with machine learning –particularly deep learning, which is currently the subject of the majority of research– one would hope to outperform systems that rely entirely on deep learning on issues such as out-of-vocabulary handling, generalizable training from small datasets, error recovery, and, in general, explainability [105].

## 6 Summary

Rapid developments in AI are changing different aspects of human life. Advances both in computational power and AI algorithm design have enabled AI methods to outperform humans in an increasing number of tasks. AI has experienced decades of praise and criticism; its path has never been smooth. With the two winters that the field has experienced, after two waves of great growth and high expectations, as well as the costs that the community of researchers, corporations, start-ups, and governments have paid, it is critical for us to recognize that the current wave of high hopes and high expectations should not be taken for granted.

## Acknowledgment

This work was in part supported by the Canadian Institutes of Health Research (CIHR) Project Grant PJT-162216. The authors also wish to acknowledge valuable feedback from Ian Janzen of BC Cancer Research Institute.

## Figures (text descriptions)

### Figure 1

A mind map titled 'Artificial Intelligence' at the center, showing the influence of various fields. The fields and their sub-topics are: Mathematics (Boolean Logic, Probability, Statistics, Algorithms, Incompleteness Theorem), Philosophy (Dualism, Empiricism, Induction, Aristotle, Ramon Llull, Leonardo da Vinci, Blaise Pascal, Thomas Hobbes, Rene Descartes, Francis Bacon, David Hume), Psychology (Cognitive Psychology, Cognitive science, Intelligence Augmentation), Neuroscience (Neuron, EEG, fMRI, Brain-Machine interface, Optogenetics), Economics (Utility theory, Decision Theory, Game theory, Multi-agent systems, Operations Research, Markov Decision Processes, Reinforcement Learning), Computer Engineering (Moore's Law, Hardware accelerators, GPUs, TPUs, WSEs, Quantum Computing, Operating Systems, Programming Languages), Control theory and Cybernetics (Water clock, Steam engine governor, Thermostat, Control theory, Cybernetics, Modern Control), and Linguistics (Introduction, Outline, Read, Analyze, Proofread).

The diagram illustrates the interdisciplinary nature of Artificial Intelligence (AI) through a central node connected to eight surrounding fields:  
  Mathematics:  Boolean Logic (First-order Logic) (George Boole, Gottlob Frege), Probability (Gerolamo Cardano), Statistics (Ronald Fisher), Bayes rules, Algorithm, Incompleteness Theorem (Computability, Tractability, NP-completeness). 
  Philosophy:  Dualism, Empiricism, Induction, Aristotle, Ramon Llull, Leonardo da Vinci, Blaise Pascal, Thomas Hobbes, Rene Descartes, Francis Bacon, David Hume. 
  Psychology:  Cognitive Psychology, Cognitive science, Intelligence Augmentation. 
  Neuroscience:  Neuron, Electroencephalograph (EEG), Functional MRI (fMRI), Brain-Machine interface, Optogenetics. 
  Economics:  Utility theory, Decision Theory, Game theory, Multi-agent systems, Operations Research, Markov Decision Processes, Reinforcement Learning. 
  Computer Engineering:  Moore's Law (Performances doubled every 18 months, Alan Turing, World War II, Deciphering codes, First Operational Computer, Konrad Zuse, First High-level Programming language, First Programmable Computer, Floating points invention, John Mauchly, J.Presper Eckert, First Electronic Computer, Charles Babbage, First Universal Computation machine), Hardware accelerators (Multi-core CPUs, GPUs (graphics processing unit), TPUs (tensor processing units), WSEs (wafer scale engines), Quantum Computing), Operating Systems, Programming Languages, Computer Software. 
  Control theory and Cybernetics:  250 BCE (First self-controlling machine, Water clock constant flow control), James Watt (18th century) (Steam engine governor, Self-regulating feedback control), Cornelis Drebbel (16th century) (Thermostat), J. C. Maxwell, Norbert Wiener (Control theory, Cybernetics), Modern Control (Stochastic optimal control, Maximize cost function over time). 
  Linguistics:  Introduction, Basic Plot, Characters, Conclusion, Outline, Read, Analyze, Drafts, Proofread. 
  A mind map titled 'Artificial Intelligence' at the center, showing the influence of various fields. The fields and their sub-topics are: Mathematics (Boolean Logic, Probability, Statistics, Algorithms, Incompleteness Theorem), Philosophy (Dualism, Empiricism, Induction, Aristotle, Ramon Llull, Leonardo da Vinci, Blaise Pascal, Thomas Hobbes, Rene Descartes, Francis Bacon, David Hume), Psychology (Cognitive Psychology, Cognitive science, Intelligence Augmentation), Neuroscience (Neuron, EEG, fMRI, Brain-Machine interface, Optogenetics), Economics (Utility theory, Decision Theory, Game theory, Multi-agent systems, Operations Research, Markov Decision Processes, Reinforcement Learning), Computer Engineering (Moore's Law, Hardware accelerators, GPUs, TPUs, WSEs, Quantum Computing, Operating Systems, Programming Languages), Control theory and Cybernetics (Water clock, Steam engine governor, Thermostat, Control theory, Cybernetics, Modern Control), and Linguistics (Introduction, Outline, Read, Analyze, Proofread).

### Figure 2

Figure 2: Summary of two-dimensional AI approaches as proposed by Russel and Norvig. The diagram shows AI branching into four approaches: Acting Rationally, Thinking Rationally, Acting Humanly, and Thinking Humanly. Each approach is further detailed with its theoretical basis and associated fields or tests.

graph LR
    AI[AI] --&gt; AR[Acting Rationally]
    AI --&gt; TR[Thinking Rationally]
    AI --&gt; AH[Acting Humanly]
    AI --&gt; TH[Thinking Humanly]

    AR --- AR_desc["The Rational Agent approach"]
    AR_desc --- AR_desc1["Rational Behavior 'doing the right thing'"]
    AR_desc1 --- AR_desc2["'the standard model' optimization problem"]
    AR_desc2 --- AR_desc3["Control Theory, Operation Research, Statistics, Economics"]

    TR --- TR_desc["The Laws of Thought approach"]
    TR_desc --- TR_desc1["Logic, Probability Theory"]
    TR_desc1 --- TR_desc2["Aristotle"]

    AH --- AH_desc["The Imitation Game 'Can a machine think?'"]
    AH_desc --- AH_desc1["Alan Turing"]
    AH_desc1 --- AH_desc2["Turing Test"]
    AH_desc2 --- AH_desc3["Natural language processing, Knowledge representation, Automated reasoning, Machine learning"]
    AH_desc2 --- AH_desc4["Extended Turing Test"]
    AH_desc4 --- AH_desc5["Computer vision, speech recognition, Robotics"]

    TH --- TH_desc["Cognitive Modelling approach"]
    TH_desc --- TH_desc1["Newel and Simon"]
    TH_desc1 --- TH_desc2["'General Problem Solver'"]
    TH_desc2 --- TH_desc3["Cognitive science, psychological experiments, brain imaging"]
    Figure 2: Summary of two-dimensional AI approaches as proposed by Russel and Norvig. The diagram shows AI branching into four approaches: Acting Rationally, Thinking Rationally, Acting Humanly, and Thinking Humanly. Each approach is further detailed with its theoretical basis and associated fields or tests.

### Figure 3

Figure 3: A timeline diagram showing the history of AI divided into three recurring cycles (hype cycles). The timeline starts in 1956 with the Dartmouth Conference (The Birth of AI) and the First Summer of AI (AI as Search Algorithms). This is followed by the First Winter of AI in 1973, marked by a Lighthill report. The Second Tide (hype cycle) begins in 1982 with the Second Summer of AI (Expert Systems era), leading to Commercialized Expert Systems. This is followed by the Second Winter of AI in 1987, marked by a Global funding cut. The Third Tide (hype cycle) begins in 1997 with the Third Summer of AI (Machine Learning era), leading to Deep Blue wins (Gary Kasparov) and AlphaNet wins (ImageNet ILSVRC) in 2012. The timeline ends in 2021 with the Third Summer cont. (Deep Learning era), marked by 'We are here now!' and three question marks pointing to the future.

Figure 3: A timeline diagram showing the history of AI divided into three recurring cycles (hype cycles). The timeline starts in 1956 with the Dartmouth Conference (The Birth of AI) and the First Summer of AI (AI as Search Algorithms). This is followed by the First Winter of AI in 1973, marked by a Lighthill report. The Second Tide (hype cycle) begins in 1982 with the Second Summer of AI (Expert Systems era), leading to Commercialized Expert Systems. This is followed by the Second Winter of AI in 1987, marked by a Global funding cut. The Third Tide (hype cycle) begins in 1997 with the Third Summer of AI (Machine Learning era), leading to Deep Blue wins (Gary Kasparov) and AlphaNet wins (ImageNet ILSVRC) in 2012. The timeline ends in 2021 with the Third Summer cont. (Deep Learning era), marked by 'We are here now!' and three question marks pointing to the future.

### Figure 4

Figure 6: (a) Schematic of the Turing Test. (b) Alan Turing.

Figure 6 consists of two parts. Part (a) is a schematic diagram of the Turing Test. It shows three main components: a 'Computer' (represented by a monitor and keyboard icon), a 'Human' (represented by a person icon with a red headband), and an 'Evaluator' (represented by a person icon with a red headband). The Computer and Human are separated by a vertical line. Below them, a horizontal line with two small document icons separates them from the Evaluator. Dashed arrows point from the Computer and Human towards the Evaluator, indicating the flow of information. Part (b) is a black and white portrait of Alan Turing, showing him from the chest up, wearing a suit and tie.  Figure 6: (a) Schematic of the Turing Test. (b) Alan Turing.

### Figure 5

A detailed horizontal timeline of AI developments from 1942 to 2021, divided into four main eras: AI Timeline (1942-1956), First Summer of AI (1956-1966), Expert Systems Era (1967-1986), and The Second Winter arrives (1987-present).

The timeline is organized into four horizontal segments, each representing a different era of AI development:  
  AI Timeline (1942 - 1956): 
 
 1942: Isaac Asimov - The idea of "Intelligent Machine" was born. 
 1943: Warren McCulloch - Water Pills - The first recognized work as AI. 
 1949: Donald Hebb - (Hebbian learning) The first updating rule for neural networks. 
 1950: Alan Turing - Turing test: "Can a Machine Think?" 
 1951: Marvin Minsky - Dean Edmonds - The first Neural Network Computer. 
 1952: Christopher Strachey - Arthur Samuel - The first AI Games. 
 
 
  First Summer of AI (1956 - 1966): 
 
 1956: John McCarthy - Marvin Minsky - Claude Shannon - Nathaniel Rochester - the term "AI" was born. 
 1956: Newell - Simon - The first "thinking humanly" approach-based program. 
 1956: Arthur Samuel - The first Reinforcement Learning program. 
 1957: Frank Rosenblatt - Mark I Perception - The first Neural Net Computer. 
 1958: John McCarthy - First High-level AI programming Lang. (LISP). 
 1959: Herbert Gelernter - First Mathematical Prover program. 
 1961: "Lispmate" - The first Industrial Robot. 
 1962: Bernard Widrow - Frank Rosenblatt - Perception Convergence Theorem. 
 1963: Marvin Minsky - The invention of "MicroWorld". 
 1965: Joseph Weizenbaum - "ELIZA" The first chatbot. 
 1966: "Shakey" - The first general-purpose mobile robot. 
 
 
  Expert Systems Era (1967 - 1986): 
 
 1969: Ed Feigenbaum - Bruce Buchanan - Joshua Lederberg - "DENDRAL" - first successful knowledge-intensive system. 
 1971: Marvin Minsky - Limitations of Perceptron. 
 1971: Bruce Buchanan - Ed Feigenbaum - Joshua Lederberg - "MYCIN". 
 1971: Carl Hewitt - "Planner" Language. 
 1972: Alan Colmerauer - Robert Kowalski - "Prolog" Language. 
 1973: "WABOT-1" - The first full-scale anthropomorphic robot. 
 1975: Emerging Hidden Markov Models (HMMs). 
 1980: K. Fukushima - "Neocogitron" - The origin of Deep Convolutional Neural Networks. 
 1982: McDermott - "R1" - first commercial expert system. 
 1986: Reinforcement of back-propagation - "connectionist models". 
 
 
  The Second Winter arrives (1987 - present): 
 
 1987: The Fifth Generation of AI in Japan and Europe. 
 1995: Yann LeCun - Deep Convolutional Neural Networks Revisited. 
 1997: Deep Blue defeated Garry Kasparov - IBM's "Deep Blue". 
 1998: Judea Pearl - Bayesian Networks in AI. 
 2000: Richard S. Sutton - Revised Reinforcement Learning. 
 2009: Fei Fei Li - Introduced ImageNet Dataset in CVPR 09. 
 2011: IBM's Watson - Watson wins Jeopardy! quiz show. 
 2012: Geoffrey Hinton - AlexNet wins ILSVRC challenge. 
 2016: DeepMind - ALPHA GO wins "Go" World Champion. 
 2020: OpenAI - Generative Pre-trained Transformer 3 (GPT-3) developed. 
 2021: Neuralink - Brain-Computer Interface chip tested. 
 
 
  A detailed horizontal timeline of AI developments from 1942 to 2021, divided into four main eras: AI Timeline (1942-1956), First Summer of AI (1956-1966), Expert Systems Era (1967-1986), and The Second Winter arrives (1987-present).

### Figure 6

Figure 9: Trade-off between interpretability and performance for AI models. The graph shows a negative correlation between model accuracy (y-axis) and model interpretability (x-axis).

The figure is a scatter plot with circles of varying sizes representing different AI models. The y-axis is labeled 'Model accuracy' and ranges from 'Low' at the bottom to 'High' at the top. The x-axis is labeled 'Model interpretability' and ranges from 'Low' on the left to 'High' on the right. The models are arranged in a descending curve from top-left to bottom-right, indicating that as interpretability increases, model accuracy generally decreases. The models, from highest accuracy/lowest interpretability to lowest accuracy/highest interpretability, are: Deep Learning, Ensembling, SVMs, Bayesian Models, Generalized Additive Models, kNN, Decision Trees, Linear / Logistic Regressions, and Rule-based Learning. The circles for Deep Learning and Ensembling are the smallest, while the circles for Generalized Additive Models and Linear / Logistic Regressions are the largest.  Figure 9: Trade-off between interpretability and performance for AI models. The graph shows a negative correlation between model accuracy (y-axis) and model interpretability (x-axis).

### Figure 7

Figure 10: Co-citation network of the 100 most cited authors with 'Artificial Intelligence' in the title. The network is divided into two main clusters: 'Connectionist' (orange nodes) and 'Symbolic' (blue nodes).

The diagram illustrates a co-citation network of 100 authors. The nodes are arranged into two primary clusters, each with a central core of highly connected authors and a periphery of more isolated nodes.  
  Connectionist Cluster (Orange):  Located on the left side of the diagram. The central core includes authors like Zadeh, Dubois, Yager, Mamdani, Kosko, Zhang, Goldberg, Holland, Karaboga, Michalewicz, Garey, Glover, Kirkpatrick, Dorigo, Kennedy, Melit, Li, Kisi, Jang, Hornik, Chau, Wang, Koza, Haykin, Hagan, Jain, Kohonen, Hopfield, Kim, Vapnik, Bishop, Rumelhart, Han, Breiman, Lecun, Hinton, Sutton, Pawlak, Quinlan, Duda, Mitchell, Michalski, Klopman, Aamodt, and Kolodner. The label 'Connectionist' is placed to the left of this cluster. 
  Symbolic Cluster (Blue):  Located on the right side of the diagram. The central core includes authors like Russell, Pearl, Dechter, Russell Stuart, Fox, Woodridge, McDermott, Nilsson, Jennings, Fikes, Kowalski, Shoham, Reiter, Rich, Lenat, Genesereth, Allen, McCarthy, Maes, Brachman, Cohen, Kuipers, Winston, Sowa, Davis, Forbus, Dekleer, Buchanan, Shortliffe, Chandrasekaran, Glancey, Felgenbaum, Schank, Winograd, Minsky, Simon, Boden, Marr, Anderson, Searle, Fodor, Dennett, Dreyfus, Turing, Newell, and Laird. The label 'Symbolic' is placed below this cluster. 
  Peripheral Nodes:  A few nodes are located at the bottom, including Shannon, Wiener, and Heckerman, who appears to be a bridge between the two clusters. 
  The network shows a clear separation between the two schools of thought, with many internal connections within each cluster and fewer connections between them.  Figure 10: Co-citation network of the 100 most cited authors with 'Artificial Intelligence' in the title. The network is divided into two main clusters: 'Connectionist' (orange nodes) and 'Symbolic' (blue nodes).

## References

1. Catherine Clifford. The ‘oracle of a.i.’: These 4 kinds of jobs won’t be replaced by robots.  https://www.cnbc.com/2019/01/14/the-oracle-of-ai-these-kinds-of-jobs-will-not-be-replaced-by-robots-.html , January 2019. Accessed: 2021-6-29.

2. Catherine Clifford. Google ceo: A.i. is more important than fire or electricity.  https://www.cnbc.com/2018/02/01/google-ceo-sundar-pichai-ai-is-more-important-than-fire-electricity.html , February 2018. Accessed: 2021-6-29.

3. Shana Lynch. Andrew ng: Why ai is the new electricity.  https://www.gsb.stanford.edu/insights/andrew-ng-why-ai-new-electricity , 2017. Accessed: 2021-6-29.

4. BBC News. Deepfake queen to deliver channel 4 christmas message.  https://www.bbc.com/news/technology-55424730#:~:text=While%20the%20Queen%20is%20delivering,news%20in%20the%20digital%20age.,  December 2020. Accessed: 2021-5-17.

5. James Vincent. Tom cruise deepfake creator says public shouldn’t be worried about ‘one-click fakes’.  https://www.theverge.com/2021/3/5/22314980/tom-cruise-deepfake-tiktok-videos-ai-impersonator-chris-ume-miles-fisher , March 2021. Accessed: 2021-5-25.

6. Oxford languages and google - english : Artificial intelligence definition.  https://languages.oup.com/google-dictionary-en/ , May 2020. Accessed: 2021-4-14.

7. Michael Aaron Dennis. Marvin minskey, american scientist: Encyclopedia britannica.  https://www.britannica.com/biography/Marvin-Lee-Minsky , January 2021. Accessed: 2021-6-29.

8. IBM Cloud Education. What is artificial intelligence (AI)?  https://www.ibm.com/cloud/learn/what-is-artificial-intelligence . Accessed: 2021-4-14.

9. Michael Chui, Martin Harrysson, James Manyika, Roger Roberts, Rita Chung, Pieter Nel, and Ashley Van Heteren. Applying AI for social good | mc kinsey et al. Technical report.

10. S.J. Russell and P. Norvig.  Artificial Intelligence: A Modern Approach . Pearson series in artificial intelligence. Pearson education limited., 2021.

11. Stuart Jonathan Russell and Peter Norvig.  Artificial Intelligence: A Modern Approach . Prentice Hall, 1995.

12. A M Turing. I.—computing machinery and intelligence.  Mind , LIX(236):433–460, October 1950.

13. Jose Hernandez-Orallo. Beyond the turing test.  J. Log. Lang. Inf. , 9(4):447–466, 2000.

14. David L Dowe and Alan R Hajek. A computational extension to the turing test. In  Proceedings of the 4th conference of the Australasian cognitive science society, University of Newcastle, NSW, Australia , volume 1. Citeseer, 1997.

15. Patrick Hayes and Kenneth Ford. Turing test considered harmful. In  IJCAI (1) , pages 972–977. researchgate.net, 1995.

16. Michael Haenlein and Andreas Kaplan. A brief history of artificial intelligence: On the past, present, and future of artificial intelligence.  California management review , 61(4):5–14, August 2019.

17. Warren S McCulloch and Walter Pitts. A logical calculus of the ideas immanent in nervous activity.  The bulletin of mathematical biophysics , 5(4):115–133, December 1943.

18. Jeremy M. Norman. McCulloch &amp; pitts publish the first mathematical model of a neural network.  https://www.historyofinformation.com/detail.php?entryid=782 . Accessed: 2021-6-7.

19. Charles Wallis. History of the perceptron.  https://web.csulb.edu/~cwallis/artificialn/History.htm . Accessed: 2021-6-7.

20. Eric Roberts. Neural networks - history.  https://cs.stanford.edu/people/eroberts/courses/soco/projects/neural-networks/History/history1.html . Accessed: 2021-6-7.

21. Gualtiero Piccinini. The first computational theory of mind and brain: A close look at mcculloch and pitts’s “logical calculus of ideas immanent in nervous activity”.  Synthese , 141(2):175–215, August 2004.

22. I, robot by isaac asimov.  https://prezi.com/r1go_sui-af2/i-robot-by-isaac-asimov/ . Accessed: 2021-4-24.

23. Alan Mathison Turing. On computable numbers, with an application to the entscheidungsproblem.  Proceedings of the London mathematical society , 2(1):230–265, 1937.

24. Donald Olding Hebb.  The organization of behavior: A neuropsychological theory . Psychology Press, 2005.

25. S Song, K D Miller, and L F Abbott. Competitive hebbian learning through spike-timing-dependent synaptic plasticity.  Nat. Neurosci. , 3(9):919–926, September 2000.

26. Jeremy Bernstein. Marvin minsky’s vision of the future.  https://www.newyorker.com/magazine/1981/12/14/a-i , 1981. Accessed: 2021-6-29.

27. Jef Akst. Machine, learning, 1951.  https://www.the-scientist.com/foundations/machine--learning--1951-65792 , May 2019. Accessed: 2021-4-25.

28. A brief history of artificial intelligence.  https://cyfuture.com/blog/history-of-artificial-intelligence/ , April 2020. Accessed: 2021-4-24.

29. Daniel Crevier.  Ai . Basic Books, August 1994.

30. Allen Newell, John C Shaw, and Herbert A Simon. Report on a general problem solving program. In  IFIP congress , volume 256, page 64. Pittsburgh, PA, 1959.

31. Arthur L Samuel. Some studies in machine learning using the game of checkers.  IBM Journal of research and development , 3(3):210–229, 1959.

32. Chris Bleakley.  Poems That Solve Puzzles: The History and Science of Algorithms . Oxford University Press, August 2020.

33. Gerald Tesauro. Temporal difference learning and TD-Gammon.  Commun. ACM , 38(3):58–68, March 1995.

34. David Silver, Aja Huang, Chris J Maddison, Arthur Guez, Laurent Sifre, George van den Driessche, Julian Schrittwieser, Ioannis Antonoglou, Veda Panneershelvam, Marc Lanctot, Sander Dieleman, Dominik Grewe, John Nham, Nal Kalchbrenner, Ilya Sutskever, Timothy Lillicrap, Madeleine Leach, Koray Kavukcuoglu, Thore Graepel, and Demis Hassabis. Mastering the game of go with deep neural networks and tree search.  Nature , 529(7587):484–489, January 2016.

35. Charles C Tappert. Who is the father of deep learning? In  2019 International Conference on Computational Science and Computational Intelligence (CSCI) , pages 343–348. ieeexplore.ieee.org, December 2019.

36. Frank Rosenblatt.  The perceptron, a perceiving and recognizing automaton Project Para . Cornell Aeronautical Laboratory, 1957.

37. Professor’s perceptron paved the way for ai – 60 years too soon.  https://news.cornell.edu/stories/2019/09/professors-perceptron-paved-way-ai-60-years-too-soon , September 2019. Accessed: 2021-4-26.

38. One page schoolhouse. Perceptron.  https://ronkowitz.blogspot.com/2017/11/perceptron.html . Accessed: 2021-4-29.

39. John McCarthy and Others.  Programs with common sense . RLE and MIT computation center, 1960.

40. Kenneth De Jong, David B Fogel, and Hans-Paul Schwefel. A2. 3 a history of evolutionary computation.  AI. 1 Introduction .

41. R M Friedberg. A learning machine: Part i.  IBM Journal of Research and Development , 2(1):2–13, 1958.

42. H Gelernter, J R Hansen, and D W Loveland. Empirical explorations of the geometry theorem machine. In  Papers presented at the May 3-5, 1960, western joint IRE-AIEE-ACM computer conference , IRE-AIEE-ACM '60 (Western), pages 143–149, New York, NY, USA, May 1960. Association for Computing Machinery.

43. Shimon Y Nof.  Handbook of Industrial Robotics . John Wiley &amp; Sons, March 1999.

44. Bernard Widrow and Others.  Adaptive adaline Neuron Using Chemical memristors . 1960.

45. Marvin Minsky.  Society Of Mind . Simon and Schuster, March 1988.

46. James R Slagle. A heuristic program that solves symbolic integration problems in freshman calculus.  Journal of the ACM (JACM) , 10(4):507–520, October 1963.

47. Thomas G Evans. A heuristic program to solve geometric-analogy problems. In  Proceedings of the April 21-23, 1964, spring joint computer conference , AFIPS '64 (Spring), pages 327–338, New York, NY, USA, April 1964. Association for Computing Machinery.

48. Daniel G Bobrow. Natural language input for a computer problem solving system. 1964.

49. Manisha Salecha. Story of eliza, the first chatbot developed in 1966.  https://analyticsindiamag.com/story-eliza-first-chatbot-developed-1966/ , October 2016. Accessed: 2021-4-26.

50. Shakey.  http://www.ai.sri.com/shakey/ . Accessed: 2021-5-4.

51. Sergei Nirenburg, Harold L Somers, and Yorick A Wilks. Alpac: the (in) famous report.  Readings in machine translation , 14:131–135, 2003.

52. Cambridge University. Sir James Lighthill FRS Lucasian Professor of Applied Mathematics. Artificial intelligence: A general survey. (lighthill report).  http://www.chilton-computing.org.uk/inf/literature/reports/lighthill_report/p001.htm , 1972. Accessed: 2021-5-4.

53. Edward A Feigenbaum, Bruce G Buchanan, and Joshua Lederberg. On generality and problem solving: A case study using the dendlal program. 1970.

54. Edward H Shortliffe and Bruce G Buchanan. A model of inexact reasoning in medicine.  Math. Biosci. , 23(3):351–379, April 1975.

55. Kunihiko Fukushima. Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position.  Biol. Cybern. , 36(4):193–202, April 1980.

56. Drew McDermott, M Mitchell Waldrop, B Chandrasekaran, John McDermott, and Roger Schank. The dark ages of ai: A panel discussion at aaai-84.  AI Magazine , 6(3):122–122, September 1985.

57. Kenneth Olsen and Harlan Anderson. Digital equipment corporation.  First People , 25, 1983.

58. David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors.  Nature , 323(6088):533–536, October 1986.

59. Goodfellow Ian, Bengio Yoshua, and Courville Aaron.  Deep Learning . MIT Press, 2016.

60. Leonard E Baum and Ted Petrie. Statistical inference for probabilistic functions of finite state markov chains.  aoms , 37(6):1554–1563, December 1966.

61. Peter W Frey and David J Slate. Letter recognition using holland-style adaptive classifiers.  Machine learning , 6(2):161–182, March 1991.

62. A Georghiades, P Belhumeur, and D Kriegman. Yale face database.  Center for computational Vision and Control at Yale University , 2(6):33, 1997.

63. Y Lecun, L Bottou, Y Bengio, and P Haffner. Gradient-based learning applied to document recognition.  Proceedings of the IEEE , 86(11):2278–2324, November 1998.

64. Christos Dimitrakakis and Samy Bengio. Online policy adaptation for ensemble algorithms. Technical report, IDIAP, 2002.

65. Mark Fanty and Ronald Cole. Spoken letter recognition.  Adv. Neural Inf. Process. Syst. , 3:220–226, 1990.

66. Victor Zue, Stephanie Seneff, and James Glass. Speech database development at mit: Timit and beyond.  Speech communication , 9(4):351–356, August 1990.

67. G H Pettengill, P G Ford, W T Johnson, R K Raney, and L A Soderblom. Magellan: radar performance and data products.  Science , 252(5003):260–265, April 1991.

68. Jinyan Li, Guozhu Dong, Kotagiri Ramamohanarao, and Limsoon Wong. Deeps: A new instance-based lazy discovery and classification system.  Machine learning , 54(2):99–124, February 2004.

69. Lester Ingber. Statistical mechanics of neocortical interactions: Canonical momenta indicatorsof electroencephalography.  Physical Review E , 55(4):4578–4593, April 1997.

70. W Nick Street, W H Wolberg, and O L Mangasarian. Nuclear feature extraction for breast tumor diagnosis. In  Biomedical Image Processing and Biomedical Visualization , volume 1905, pages 861–870. International Society for Optics and Photonics, July 1993.

71. Zi-Quan Hong and Jing-Yu Yang. Optimal discriminant plane for a small number of samples and design method of classifier on the plane.  Pattern Recognition , 24(4):317–324, January 1991.

72. A M Bagirov, A M Rubinov, N V Soukhoroukova, and J Yearwood. Unsupervised and supervised data classification via nonsmooth and global optimization.  TOP , 11(1):1–75, June 2003.

73. J R Quinlan, P J Compton, K A Horn, and L Lazarus. Inductive knowledge acquisition: a case study. In  Proceedings of the Second Australian Conference on Applications of expert systems , pages 137–156, USA, October 1987. Addison-Wesley Longman Publishing Co., Inc.

74. David Clark, Zoltan Schreter, and Anthony Adams. A quantitative comparison of distal and backpropagation. In  Australian conference on neural networks , 1996.

75. Wayne Iba, James Wogulis, and Pat Langley. Trading off simplicity and coverage in incremental concept learning. In John Laird, editor,  Machine Learning Proceedings 1988 , pages 73–79. Elsevier, San Francisco (CA), January 1988.

76. Judea Pearl.  Causality . Cambridge University Press, September 2009.

77. Richard S Sutton and Andrew G Barto.  Reinforcement learning: An introduction . MIT press, 2018.

78. Bruce Weber. Computer defeats kasparov, stunning the chess experts.  The New York Times , May 1997.

79. Chris Higgins. A brief history of deep blue, IBM’s chess computer.  https://www.mentalfloss.com/article/503178/brief-history-deep-blue-ibms-chess-computer , July 2017. Accessed: 2021-5-3.

80. Michael A Morris, Babak Saboury, Brian Burkett, Jackson Gao, and Eliot L Siegel. Reinventing radiology: Big data and the future of medical imaging.  Journal of thoracic imaging , 33(1):4–16, January 2018.

81. Dave Gershgorn. The data that transformed ai research—and possibly the world.  https://qz.com/1034972/the-data-that-changed-the-direction-of-ai-research-and-possibly-the-world/ , July 2017. Accessed: 2021-5-4.

82. Adam Gabbatt. Ibm computer watson wins jeopardy clash.  https://www.theguardian.com/technology/2011/feb/17/ibm-computer-watson-wins-jeopardy , February 2011. Accessed: 2021-5-17.

83. Yann LeCun, Bernhard Boser, John S Denker, Donnie Henderson, Richard E Howard, Wayne Hubbard, and Lawrence D Jackel. Backpropagation applied to handwritten zip code recognition.  Neural computation , 1(4):541–551, 1989.

84. Imagenet large scale visual recognition competition 2012 (ilsvrc2012).  https://image-net.org/challenges/LSVRC/2012/results.html . Accessed: 2021-5-4.

85. Kumar Chellapilla, Sidd Puri, and Patrice Simard. High performance convolutional neural networks for document processing. In  Tenth International Workshop on Frontiers in Handwriting Recognition . hal.inria.fr, 2006.

86. Dan Claudiu Ciresan, Ueli Meier, Jonathan Masci, Luca Maria Gambardella, and Jürgen Schmidhuber. Flexible, high performance convolutional neural networks for image classification. In  Twenty-second international joint conference on artificial intelligence . people.idsia.ch, 2011.

87. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks.  Adv. Neural Inf. Process. Syst. , 25:1097–1105, 2012.

88. Xiaoxuan Liu, Livia Faes, Aditya U Kale, Siegfried K Wagner, Dun Jack Fu, Alice Bruynseels, Thushika Mahendiran, Gabriella Moraes, Mohith Shamdas, Christoph Kern, Joseph R Ledsam, Martin K Schmid, Konstantinos Balaskas, Eric J Topol, Lucas M Bachmann, Pearse A Keane, and Alastair K Denniston. A comparison of deep learning performance against health-care professionals in detecting diseases from medical imaging: a systematic review and meta-analysis.  The Lancet Digit Health , 1(6):e271–e297, October 2019.

89. Mark Manson. I, for one, welcome our AI overlords.  https://markmanson.net/artificial-intelligence , June 2016. Accessed: 2021-4-14.

90. Raymond Perrault, Yoav Shoham, Erik Brynjolfsson, Jack Clark, John Etchemendy, Barbara Grosz, Terah Lyons, James Manyika, Saurabh Mishra, and Juan Carlos Niebles. The AI index 2019 annual report. Technical report, , AI Index Steering Committee, Human-Centered AI Institute, Stanford University, Stanford, CA, December 2019.

91. Daniel Zhang, Saurabh Mishra, Erik Brynjolfsson, John Etchemendy, Deep Ganguli, Barbara Grosz, Terah Lyons, James Manyika, Juan Carlos Niebles, Michael Sellitto, et al. The ai index 2021 annual report.  arXiv preprint arXiv:2103.06312 , 2021.

92. Michael Chui, Martin Harrysson, James Manyika, Roger Roberts, Rita Chung, Pieter Nel, and Ashley van Heteren. Applying artificial intelligence for social good.  https://www.mckinsey.com/featured-insights/artificial-intelligence/applying-artificial-intelligence-for-social-good , November 2018. Accessed: 2021-4-14.

93. Tao Ruspoli. Being in the world - on the subject of the #heideggerian dasein, April 2018. Alive Mind Cinema.

94. Timothy Taylor. 1957: When machines that think, learn, and create arrived.  https://www.bbntimes.com/global-economy/1957-when-machines-that-think-learn-and-create-arrived . Accessed: 2021-6-29.

95. James Vincent. Forty percent of ‘ai startups’ in europe don’t actually use AI, claims report.  https://www.theverge.com/2019/3/5/18251326/ai-startups-europe-fake-40-percent-mm-c-report , March 2019. Accessed: 2021-4-17.

96. Zachary C Lipton and Jacob Steinhardt. Research for practice: troubling trends in machine-learning scholarship.  Communications of the ACM , 62(6):45–53, May 2019.

97. Michael Roberts, Derek Driggs, Matthew Thorpe, Julian Gilbey, Michael Yeung, Stephan Ursprung, Angelica I Aviles-Rivero, Christian Etmann, Cathal McCague, Lucian Beer, Jonathan R Weir-McCall, Zhongzhao Teng, Effrossyni Gkrani-Klotsas, James H F Rudd, Evis Sala, and Carola-Bibiane Schönlieb. Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans.  Nature Machine Intelligence , 3(3):199–217, March 2021.

98. Irving Wladawsky-Berger. What machine learning can and cannot do.  https://www.wsj.com/articles/what-machine-learning-can-and-cannot-do-1532714166?tesla=y , July 2018. Accessed: 2021-4-18.

99. AI set to exceed human brain power.  http://edition.cnn.com/2006/TECH/science/07/24/ai.bostrom/ , August 2006. Accessed: 2021-4-29.

100. Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners.  arXiv e-prints , May 2020.

101. Luciano Floridi. Establishing the rules for building trustworthy ai.  Nature Machine Intelligence , 1(6):261–262, May 2019.

102. David Gunning and David Aha. Darpa’s explainable artificial intelligence (xai) program.  AI Magazine , 40(2):44–58, June 2019.

103. Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, Adrien Bennetot, Siham Tabik, Alberto Barbado, Salvador Garcia, Sergio Gil-Lopez, Daniel Molina, Richard Benjamins, Raja Chatila, and Francisco Herrera. Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai.  Information Fusion , 58:82–115, June 2020.

104. David Gunning. Explainable artificial intelligence (xai).  Defense Advanced Research Projects Agency (DARPA), nd Web , 2(2), 2017.

105. Kamruzzaman Sarker, Lu Zhou, Aaron Eberhart, and Pascal Hitzler. Neuro-symbolic artificial intelligence: Current trends.  arXiv e-prints , page arXiv:2105.05330, May 2021.
