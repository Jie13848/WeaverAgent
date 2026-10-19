# Avalon: A Benchmark for RL Generalization Using Procedurally Generated Worlds

```meta
arxiv: 2210.13417
```

## Authors

- Joshua Albrecht   Abraham J. Fetterman   Bryden Fogelman   Ellie Kitanidis
- Bartosz Wróblewski   Nicole Seo   Michael Rosenthal   Maksis Knutins
- Zachary Polizzi   James B. Simon   Kanjun Qiu
- Generally Intelligent*

## Abstract

Despite impressive successes, deep reinforcement learning (RL) systems still fall short of human performance on generalization to new tasks and environments that differ from their training. As a benchmark tailored for studying RL generalization, we introduce Avalon, a set of tasks in which embodied agents in highly diverse procedural 3D worlds must survive by navigating terrain, hunting or gathering food, and avoiding hazards. Avalon is unique among existing RL benchmarks in that the reward function, world dynamics, and action space are the same for every task, with tasks differentiated solely by altering the environment; its 20 tasks, ranging in complexity from  eat  and  throw  to  hunt  and  navigate , each create worlds in which the agent must perform specific skills in order to survive. This setup enables investigations of generalization within tasks, between tasks, and to compositional tasks that require combining skills learned from previous tasks. Avalon includes a highly efficient simulator, a library of baselines, and a benchmark with scoring metrics evaluated against hundreds of hours of human performance, all of which are open-source and publicly available. We find that standard RL baselines make progress on most tasks but are still far from human performance, suggesting Avalon is challenging enough to advance the quest for generalizable RL.

## 1 Introduction

A central goal of reinforcement learning (RL) is to build systems that can master a spectrum of skills in environments as noisy and diverse as the real world. While RL algorithms have matched or exceeded human performance on a number of narrowly-defined tasks such as Go and Atari [4, 7, 28, 34, 40], existing models typically fail to generalize to unseen tasks and environments, even when testing on environments drawn from the same distribution as training [11, 45, 46]. The real-world setting is far more diverse and difficult than most existing RL benchmarks; agents must seamlessly interact in a highly variable 3D environment, with no access to state information or ability to rely on hard-coded discrete actions, requiring sensory inputs and a high-dimensional action space. *Correspondence to  avalon@generallyintelligent.com . Figure 1:  Avalon is an open-world survival game requiring an agent to navigate hazards and find food.  Top: aerial view of a procedurally-generated world. Bottom row, left to right: examples of fruit to consume, predators to avoid, terrain to navigate, and buildings to explore. To aid the study of generalization in a more realistic setting that is still tractable for current RL algorithms, we present Avalon 2 : a benchmark and high-performance simulator. Avalon is a 3D open-world survival game in which agents must navigate terrain, find food, use tools, hunt prey, and avoid predators and other dangers. Highly diverse 3D worlds are procedurally generated with many biomes, plants, items with distinct properties, and animals with unique behaviors. In any particular world, the agent’s sole objective is to survive for as long as possible by finding and eating all food while avoiding hazards. Avalon agents are embodied and receive visual and proprioceptive input, forcing them to learn representations rather than relying on latent state information that would be inaccessible in the real world. We formulate Avalon as a multi-task RL benchmark in which each task is shaped by environmental pressures rather than a task-dependent reward function. This enables all tasks to have shared dynamics and reward, making generalization more feasible compared to disjoint task spaces such as Atari’s [3] where reduced or even negative transfer learning has been noted [31] [36]. Twenty tasks ranging in complexity from  eat  and  throw  to  hunt  and  survive  are included in the benchmark. A  world generator  for each task carefully alters a world so that certain skills are required to complete the task. For example, the  climb  generator only spawns food in places that cannot be accessed without climbing. Thus, Avalon’s tasks are effectively a set of functions for procedurally generating environments that force the agent to learn the requisite skill to survive. Avalon facilitates the exploration of a variety of forms of generalization: To unseen tasks that are structurally similar to previous tasks.  All tasks in Avalon share the same transition dynamics, action space, observation space, and simple reward based on the agent’s energy level. Environment variation between tasks and within tasks is executed through the same mechanism i.e. sampling from a subset of the full distribution of possible worlds. Indeed, one could imagine widening the scope of any task-specific world generator until it encompasses some or all other tasks. This notion of task “adjacency” enables the agent to exploit shared structure between tasks. To unseen combinations of tasks.  In addition to 16 basic generators that map to basic skills, four compositional generators create worlds that require the agent to exercise random combinations, 2 Named after a mythical land of Arthurian legend whose etymology is “the isle of fruit trees,” since Avalon’s worlds are mostly islands with fruit trees wherein the ultimate goal is to find and eat the fruit. permutations, and variations of those skills. These compositional generators can be used to evaluate the agent’s performance on unseen tasks that require many skills at once and provide a new setting to study the generalizability and compositionality of learned skills. In other multi-task setups, compositional tasks are often difficult to express. For example, multi-term reward functions quickly become cumbersome, and designing goal states that require complex sequential tasks is challenging. To unseen environments within tasks.  Unlike many popular RL benchmarks (e.g. Atari and DeepMind Control [2]) where the test environment is identical to the train environment, Avalon has significantly more variation between worlds sampled from the same distribution. Even compared to other procedurally generated benchmarks like ProcGen [8] and MineRL [15], Avalon contains dramatically more factors of variation, each of which can be individually and finely controlled. Agents in Avalon must generalize to IID sampled test environments in a setting with significantly more realism and complexity than comparable benchmarks. Avalon’s continuous action space maps to a virtual reality (VR) headset and controllers. Our choice of observation and action space allows us to measure human performance by recording demonstrations using a VR headset. We include this dataset of human VR actions, with 215 hours of human playthroughs on 1000 worlds, which may be of independent research interest. Our benchmark includes a number of state-of-the-art RL algorithms as baselines. In Section 6, we show their performance on our tasks and highlight training mechanisms enabled by our world generation setup that help them achieve non-trivial (though still far short of human) performance. As a final contribution, we are fully open-sourcing our environment, which represents a significant engineering effort and includes the following key contributions: •  Efficiency.  At 7000 SPS (steps per second), Avalon is on-par with the fastest comparable simulator [38].  •  Usability.  Avalon is fully open-source, built on Godot (a free game engine with an intuitive visual editor), and includes training scripts and debugging tools.  •  Configurability.  Control over the many factors of variation in the world generators enables users to fully control the training procedure and learning curriculum. Our hope is that the Avalon benchmark and simulator will serve as useful tools for the RL community in the quest to build more general, capable learning systems.

## 2 Related work

Avalon is the only benchmark where 1) agents learn from high-dimensional inputs in 3D procedurally generated worlds with a continuous action space, 2) the observation space, action space, transition dynamics, and reward are held constant across all tasks, 3) a very large number of factors of variation can be finely controlled in order to isolate and explore specific types of generalization, and 4) the underlying simulator is very fast and easy to use. To achieve these goals, Avalon builds on many ideas from previous works, discussed below. Games have historically been the gold standard for benchmarking reinforcement learning methods [4, 7, 40]. The Arcade Learning Environment [3] and DeepMind Lab [2] are early examples of multi-task benchmarks that encourage mastery across several tasks. On these benchmarks, overfitting due to trajectory memorization [23] and poor or negative transfer across tasks [31, 36] remain issues. Procedurally generated benchmarks such as ProcGen [8], MineRL [15], Malmo [19], and MetaDrive [27] aim to address per-task overfitting by varying environmental factors such as the background and placement of objects and obstacles. However, most rely on a single random seed for procedural content generation or allow users to vary only a few (usually discrete) parameters. MineRL and Malmo also run at less than 50 steps per second on a single GPU (over 100 times slower than Avalon), which makes their use for large experiments slow and expensive. Other works such as Meta-World [44] attempt to address poor task transfer by unifying the observation space, action space, and transition dynamics across all tasks and adding controllable parametric variation of object and goal positions in order to increase shared structure across tasks. However, these environments rely on direct access to state information and include per-task handcrafted dense Figure 2: Example tasks  bridge ,  fight , and  stack  (left). World generators for each task shape the environment to require completion of the task to reach food. Tasks can be outdoor or inside buildings in the environment. Buildings also enable  open  (right), where doors with complex locking mechanisms must be solved to get to food. rewards, both of which limit applicability to the broad set of tasks that general agents are expected to solve. Other benchmarks such as Crafter [16], Obstacle Tower [22], and the Animal-AI Environment [5] avoid per-task rewards by defining a single reward function. These environments all have discrete action spaces and are limited in terms of diversity. Crafter is a 2D world; Obstacle Tower has few game mechanics: opening doors, picking up keys, pushing blocks; and Animal-AI tasks exist in a flat arena with a dozen object options. Several other environments have substantially more diversity. MiniHack [37] is a sandbox for designing environments in the text-based game NetHack [26], which has a randomized system of dungeons with many creatures and items. However, mastery of NetHack is elusive even for humans and requires extensive game-specific knowledge, making it ill-suited to generalization research, and NetHack’s symbolic inputs do not test an agent’s ability to learn visual representations. By contrast, XLand [29] is a 3D multi-agent environment with visual inputs and a single reward function across tasks. Agents in XLand operate in a low-dimensional, discrete action space and are provided with explicit state information about the goals and game predicates, which severely limits the tasks that can be expressed in this constrained symbolic system. Additionally, XLand is not publicly released, so it cannot be used as a benchmark. A separate class of related work includes photorealistic embodied AI environments such as Habitat Lab [38], GibsonEnv [43], Megaverse [32], RFUniverse [12] and BEHAVIOR [41], 3D simulators such as ThreeDWorld [13], and dataset generators such as Kubric [14]. These are valuable contributions for embodied AI and visual representation learning, but they are not designed as benchmarks for exploring RL generalization. Additionally, most of these simulators run at less than 100 steps per second (SPS) on a single GPU, which is prohibitively slow for RL research. Habitat Lab runs significantly faster (around 8,000 SPS on a single GPU), but has a limited action space, limited allowed physical interactions, and no procedural generation, making it ill-suited as a benchmark for generalization.

## 3 RL interface

The Avalon RL environment conforms to the standard OpenAI Gym environment interface [6] and is built on top of the open-source Godot game engine [21] and Bullet physics engine [9]. Below, we present some details of this environment and the design decisions that led to it.

## 3.1 Game mechanics

Avalon is a 3D open-world survival game in which players must overcome obstacles and hazards while acquiring food. Gameplay consists of a series of episodes, where each episode corresponds to a unique world, which is generated by a task-specific world generator. A key aspect of Avalon is that special subsets of the environment distribution map to distinct tasks. Environmental pressures arising from the presence and placement of various interactable objects, enemies, and even terrain features demand the execution of a broad array of navigation and object interaction skills. For example, the existence of a deep chasm between the player and the food forces the player to create a bridge, or food may be placed in a region that is inaccessible without climbing. Figure 2 shows some examples of generated worlds for a variety of tasks. Another critical part of Avalon is the diversity of gameplay mechanics and worlds that can be generated. There are 14 biomes, 13 plants, 20 interactable items, and 17 animals. Items include food, weapons, elements such as boxes for stacking, doors, and so on. Animals include both predators and prey, where prey are edible and can be consumed by the agent as food. All animals and food have completely distinct behaviors (for example, jaguars aggressively pursue the player and can climb trees, coconuts must experience a certain force to be opened, etc). See Appendix A for a complete description of each entity included in the game. While the setting of Avalon is inspired by the environment in which humans evolved, it includes a number of simplifications designed to accommodate current RL systems. For example, most objects are significantly larger than their real-life counterparts because current RL algorithms are prohibitively expensive to train with very high-resolution images. As another mechanism to allow for low-resolution agents, we guarantee that food is always found near buildings or certain large trees, both of which are visible from far away.

## 3.2 Rewards

Regardless of the task, the goal of players in Avalon is to survive for as long as possible. Both human players and RL agents have a single scalar “energy” value, and when this reaches zero, the episode ends. The only way to gain energy is to eat, and thus acquiring food is the implicit goal of every level. Energy can be lost from attacks by predators or falling too far. In order to represent these mechanics to the agent, the reward from the environment is simply calculated as the change in energy at each time step. While the reward is technically dense, in that the agent observes small changes in energy at each frame, it is effectively a sparse reward, since positive reward is attained only when the agent eats food i.e. has successfully performed the task. To encourage efficiency and prevent erratic motion, small energy costs can be associated with movement for agents during training. More details on the exact calculations for the reward function and time limits in each episode can be found in Appendix B. Though this (effectively) sparse reward setting is challenging, we note that dense rewards usually need to be tailored to each task and often rely on hidden state information. However, it is straightforward for a user to adapt Avalon to employ a dense reward if desired.

## 3.3 Observations

While the state data for our simulation is easily accessible, we refrain from providing any ground truth information in the observations that is not accessible to human players. The agent receives egocentric visual input in the form of a  96 \times 96 \times 4  RGBD tensor. Just like human players, the agent is embodied and can see its own hands in its visual display. The agent is also provided with basic proprioceptive awareness: its input includes the framewise change in position and orientation for its body, the position and orientation of each of its hands (relative to its body), boolean indicators for whether each hand is within grasping range of (or currently grasping) an object, the current value of its energy, and the number of frames remaining until the episode times out. See Appendix C for the exact list of observed variables.

## 3.4 Actions

The embodied agent can move through and interact with its environment using its head, body, and hands. The primary action space is 21-dimensional and roughly maps to a virtual reality headset and controllers. It is made of  3 \times (3 + 3)  translational plus rotational degrees of freedom, as well as  2 \times 1  binary grasp actions and 1 discrete jump action. We also provide a reduced 9-dimensional action space which corresponds to mouse and keyboard controls. Almost all tasks can be accomplished in this reduced action space. See Appendix D for the exact action space definition.

## 3.5 Simulator

Unlike many popular RL environments, Avalon is not built on top of an existing game but rather constructed from the ground up to optimize for speed, accessibility, and, above all, scientific value. This has enabled several benefits that would otherwise be impossible: • Avalon simulates roughly 7,000 SPS, orders of magnitude faster than most other simulators, and similar to Habitat-Sim, the fastest comparable simulator. The simulator runs without a display, making it even more efficient and easy to work with.  • Avalon was designed for the purposes of ML research, and thus benefits from a number of debugging capabilities. Firstly, it is deterministic; playing back the same set of actions from the same starting seed results in identical output. Secondly, the ground truth states of all objects can be logged continually, making it highly inspectable.  • Avalon is made using the Godot game engine [21], which is fully open-source and cross-platform (it runs on Windows, Linux, Mac, and several popular VR headsets). Godot is simple and lightweight, with a modest  \sim 30\text{MB}  install size, and has a clean asset pipeline that supports most common formats. It boasts a full-featured visual editor, debugger and profiler that make it very easy to work with. All of our code is written in either Python or the Python-esque Godot scripting language, so it is straightforward for researchers to modify and extend, especially given the vibrant community of professional and hobbyist developers that also use Godot. Godot is released under an MIT license, while the rest of our code is released under the GPL license.

## 4 Procedural environment generation

Avalon contains a sophisticated system for generating worlds that gives researchers fine-grained control over every aspect of variation—such as ruggedness of terrain, height of cliffs, density of predators, prey, and plants, etc.—while enabling extremely high diversity. This system empowers researchers to isolate and explore many specific aspects of generalization, ranging from the most straightforward IID setting to OOD settings like the creation of complex compositional tasks or even entirely new tasks. It also enables a simple difficulty-based curriculum that accelerates learning for our included baseline systems (described in more detail in Section 6). Avalon supports easily scaling the diversity of the training distribution, allowing for exploration of basic IID generalization, where the test world distribution matches the training distribution. In our benchmark setting, we primarily vary terrain height and geography while leaving most diversity turned off (e.g. we don’t vary colors or models of trees, landscape, food, prey, or predators, nor do we vary objects in size, shape, or weight, etc.). As RL systems become more capable, this diversity can be dramatically increased to provide more challenging IID generalization settings. To enable investigation of generalization between similar tasks, Avalon’s task-specific world generators carefully alter each procedurally generated world to require the use of particular skills. For example, the generator can raise a section of terrain to surround food with a cliff or insert an unclimbable chasm to require jumping. Our hope is that this setup enables better transfer between similar tasks, as many tasks allow for multiple solutions and the use of multiple skills, especially at lower difficulties. In order to address a more compositional form of OOD generalization, we have designed our task-specific world generators so that they can be applied to a single world to create a sequence of tasks that must be performed. For example, our “jump” generator creates a natural-looking chasm that can only be jumped across at a certain point, while our “climb” generator creates a ring of sloped terrain that can only be climbed along a certain path. By simply composing one of these obstacle “rings” inside the other, we can create a world in which the player must accomplish both. Our fine-grained control over diversity in world generation can also be used to directly ask about OOD generalization under small shifts in the distribution of environments. All variation in Avalon’s world generation is finely controllable: the sizes and colors of all objects, the number of predators, prey, food, etc in a task, the distance of a gap that must be jumped across, etc (see Appendix E for a full list). This allows for simple, continuous ablations as any individual factor is varied from the training setting, all without any need to create or import any manual art assets.

## 5.1 Tasks

Avalon consists of 20 distinct tasks testing a range of navigation and object manipulation skills in diverse environments. This list includes 16 “basic” tasks ( eat ,  move ,  jump ,  climb ,  scramble ,  descend ,  throw ,  hunt ,  fight ,  avoid ,  push ,  stack ,  bridge ,  open ,  carry , and  explore ) and four “compositional” tasks ( navigate ,  find ,  gather , and  survive ). Each task has the same agent goal — acquiring food while avoiding hazards — but the worlds generated for each task are structured such that attaining this goal requires the skill being tested. Worlds for the 16 basic tasks include a single food item that must be reached by overcoming at most a single obstacle. The  eat  task is the easiest: the food is created near the agent, which needs to merely grab it and bring it to its head. The other basic tasks build upon this in various ways. For example,  avoid  requires evading a predator to reach the food,  hunt  requires hunting moving prey, and  stack  requires stacking objects to reach food on a high ledge. Worlds for the four compositional tasks are designed to require the sequential use of several basic skills:  navigate  places a series of basic obstacles between the player and the food,  find  is the same but the food is not guaranteed to be visible from the agent’s starting location,  gather  has multiple fruits to find, and  survive  includes both fruit and prey animals.  survive  is a sort of “final exam” focused on breadth whose worlds can contain the elements of every other task. See Appendix F for a complete definition of every task.

## 5.2 Training and evaluation protocols

We outline four settings in which agents can be trained and evaluated within Avalon: •  Multi-Task, Train Basic (MT-TB) : Train on the 16 basic tasks and evaluate on all 20 tasks. In this version of the multi-task setup, agents have seen each of the intermediary tasks but have never seen them composed together until test time.  •  Multi-Task, Train All (MT-TA) : Train and evaluate on all 20 tasks.  •  Multi-Task, Train Compositional (MT-TC) : Train on the four compositional tasks and evaluate on the full set of 20 tasks.  •  Single Task, Basic (ST-B) : Train and evaluate on each of the 16 basic tasks separately. Agents are trained with an adaptive curriculum (Section 6.1) such that the number of worlds seen in a given amount of training time is variable. While we present a particular curriculum-based training procedure, users of the benchmark are encouraged to explore alternative training procedures. Agents are evaluated on a fixed set of 50 worlds for each relevant task, as indicated above. For example, the MT-CG and MT-S settings evaluate on all 20 tasks, and thus use the full set of 1,000 possible evaluation worlds. Tasks were randomly generated and not manually curated, except to replace levels that were reported as practically impossible. Upon investigation only one level was actually impossible, representing 0.1% of the originally generated levels. We also ensured that the evaluation worlds fully represented their underlying task generators. For example, the set of evaluation tasks for “ eat ” includes at least one of every possible type of food. See Appendix G for more details about the exact distribution of evaluation worlds, including the replaced worlds.

## 5.3 Human performance

To understand the difficulty of each of our tasks, we collected human player data for all evaluation levels. Approximately 215 hours of VR gameplay were recorded from 32 participants drawn from a pool of volunteers whose familiarity ranged from zero to significant experience playing similar games. For each of the  50 \times 20 = 1000  evaluation levels, scores were averaged across at least 5 different players. Humans were provided with two practice levels for each task (not in the evaluation set), as well as basic instructions about the game mechanics (see Appendix H for details on the information provided and human data collection procedure). Ground truth control data was recorded for each player on each level, and is available on our main website under a CC BY-SA license. We do not recommend behavior cloning from this data and testing on the fixed evaluation set; however, researchers can easily generate new evaluation levels from the same distribution for testing any such networks. The human data may also be interesting to inspect for trends, training effects, or unrelated research into VR human gameplay.

## 5.4 Performance metrics

Given a set of human player data, raw rewards can be converted into scores  S  by normalizing such that the average human performance is 1.0 and the performance of a random agent is 0.0. We recommend two different aggregations of these scores. The first and more traditional aggregation is to simply take the average of these scores. The second and more robust aggregation is the optimality gap [1]. By plotting the cumulative frequency of these scores  P(S &gt; x)  across all runs, the optimality gap expresses how consistently the agent achieves performance above some fraction of human performance. See Appendix M for details on how scores are calculated.

## 6.1 Difficulty curriculum

Due to the variety and scope of tasks and environments in our benchmark, we found that agents failed to learn when training on worlds sampled uniformly at random. Because the agent is effectively only given sparse positive rewards (i.e. if it finds and eats food), it is difficult for agents to make progress on more difficult tasks before improving at the  eat  and  move  tasks. For most tasks, uniform random sampling of worlds leads to a very low probability of creating worlds where random exploration policies get any reward. As one approach to overcoming these issues, we use a simple “difficulty”-based curriculum. We employ a simple form of Automated Curriculum Learning [33] similar to the approaches in [24, 30]. When the agent succeeds (fails) at a task, the maximum difficulty of future generated worlds for that task is increased (decreased) slightly. See Appendix I for more details.

## 6.2 Baselines

We trained IMPALA [10, 25], PPO [35, 39], and DreamerV2 [17, 18] on our benchmark using the training protocols outlined above, including the difficulty curriculum. We chose IMPALA and PPO as baselines because they are among the simplest SOTA algorithms on which many other SOTA algorithms are based, and tend to be robust across problems. We chose DreamerV2 as a representative model-based baseline. DreamerV2 is used without discrete latents and without mixed precision training. No worlds are seen twice during training. Hyper-parameters were tuned via a combination of Bayesian optimization [20] and Natural Evolution Strategies (NES) [42] using runs with a smaller number of steps, and are included in Appendix J. Scores are the average of 5 runs (for 50m step results) or 3 runs (for the 500m step results). See Appendix K for details about the machines used for training.

## 6.3 Results and discussion

We show average scores for the MT-TB setting in Table 1. While all algorithms are able to achieve non-zero performance on most tasks, they fall far short of human performance, and show particularly low performance when testing on the four compositional tasks. It should be noted that non-zero performance is sometimes indicative of unexpected strategies; for example, we observe agents learning to open doors by jumping while grabbing the latch rather than moving their hand to lift the latch, thus reliably opening doors without necessarily understanding doors. Table 1 also includes the results of longer training and ablating the curriculum learning component. The longer training shows results comparable to the 50m step training, indicating convergence of IMPALA, although learning curves in Appendix L indicate some tasks might still see improvement with longer training. Using no curriculum results in scores that are no better than random. Table 1: Average agent scores on all Avalon tasks. Rows show scores on basic tasks (first section), compositional tasks (second section), and aggregations of tasks (third section), normalized such that mean human performance is one. The header indicates the algorithm, the total number of environment steps and whether the training curriculum is used. Due to space constraints, the per-task optimality gap scores are reported in Appendix L (see Table 8) along with the scores from the other train-evaluate settings (MT-TA, MT-TC, ST-B) and additional results. Comparisons of the different settings (see Table 9 and Table 10) provide multiple lenses through which to view generalization. For example, we find that several tasks such as  climb ,  descend , and  avoid  are too challenging for the agent to learn in the single-task setting (ST-B), yet they can be partially mastered in the multi-task settings (MT-TB, MT-TA) after pre-training on other tasks, suggesting the shared structure of the tasks provides transfer learning benefit. Furthermore, we find that agents trained on only the 16 basic tasks (MT-TB) perform as well as or better than agents trained on all 20 tasks (MT-TA), even on the compositional tasks themselves which the agent has never seen in training under MT-TB.

## 7 Limitations

While Avalon as it is today enables targeted exploration of many types of generalization in RL, there are a number of limitations. First, given the scope of the diversity possible in our environments, there are sure to be bugs and unintended results (for example, some generated worlds are impossible to complete). Another limitation is fundamental to procedural generation; while the generated worlds are quite complex, they are still far less complex than the real world, and as such, success on Avalon should not be taken as evidence that generalization is “solved” in the general case. Finally, due to limited time, we were only able to run a small number of baseline algorithms. We intend to publish updated baselines and results as they become available.

## 8 Future work

As it exists today, Avalon enables a variety of experiments, from the generalization settings highlighted here to better curriculum designs and unsupervised skill discovery. Avalon is also highly extensible by design. Users can easily create entirely new game mechanics, enabling RL research beyond the simple navigation and object manipulation tasks we have presented here. We plan to extend Avalon in a number of ways, including by creating more tasks and more variety within each task. We also plan to include more sophisticated environment interactions to encourage tool making and tool use, as well as longer time-horizon mechanics such as rest, day-night cycles, and regrowing food to encourage the creation of RL agents that can deal with longer time horizons. Finally, we also hope to eventually include multi-agent components as well.

## 9 Conclusions

We created Avalon to address the need for a better benchmark for RL generalization and robustness. By creating a diverse set of tasks solely via environmental variation, Avalon is able to create challenging worlds that share task structure and world dynamics, hopefully encouraging the creation of more powerful and generally capable RL agents. The tasks in the Avalon benchmark are quite challenging for existing systems, despite being relatively trivial for people. Despite this, our training procedures provide a starting point with reasonable performance, on which others can improve. We hope our simulator and benchmark will serve as a foundational piece of infrastructure for future research on generalization, exploration, and other topics that are under-served by existing benchmarks.

## Acknowledgments

The authors would like to thank the following individuals for invaluable discussions and feedback on this benchmark: Anwar Bey, Tom Brown, Michael Chang, Shreyas Kapur, Andrew Lampinen, Joel Lehman, Rosanne Liu, Luke Melas-Kyriazi, John Schulman, Elias Wang, Yi-Fu Wu, and Wojciech Zaremba. The authors are also grateful for the hard work and adventurous spirit of our human Avalon players: Meera Balakumar, Akshiv Bansal, Michael Bonanni, Andrew Cote, Rob Courtice, Dane Cross, Todd Dabney, Samkeliso Dlamini, Alejandra Encinas, Lincoln Scott Fuller, Amanda Gabbara, Nick Gabbara, Maddy Gaffaney, Gjon Gjeloshaj, Perry Goldstein, Cecilia Goss, Patrick Hoon, Schuyler Howe, Asmeret Jafarzade, Keil Miller Joseph, Luke Juusola, John Lindstedt K., Yad Konrad, Cory Li, Terrence Lucero, Kathryn Martin, Annie Melton, Brian Demeyer Michael, Kevin Multani, Jeremy Nelson, Carol Ng, Sam O’Donnell, Sam Parks, Divyesh Patel, Matthias Pauthner, Dominick Pierre-Jacques, Maria Polizzi, Nathan Ravenel, Roy Rinberg, Snigdha Roy, Katie Sapko, Phillip Seo, Brian Smiley, Derek Tam, Kaspars Vandans, Helen Wei, Christina Zhu.

## Supplementary Materials: Content Overview

• Appendix A contains details about each game element.  • Appendix B contains details about the reward function and starting energy levels.  • Appendix C contains a more detailed specification of the observation space.  • Appendix D contains a more detailed specification of the action space.  • Appendix E contains an overview of the different types of factors of variation in the generated worlds, with pointers to the full documentation in the code.  • Appendix F contains a brief description of the mechanics of each of the 16 basic tasks and four compositional tasks.  • Appendix G contains more details about the exact procedure for generating and selecting the worlds used for evaluation.  • Appendix H contains a description of the information that was provided to the human participants as well as other details about the human data collection.  • Appendix I contains more details about our exact curriculum learning algorithm.  • Appendix J contains the hyperparameters that were used for each network.  • Appendix K contains more details on the exact compute hardware used for training.  • Appendix L contains additional experimental result tables and figures, including optimality gap scores, scores for all training conditions, performance breakdown by task, and learning curves.  • Appendix M contains more details on how scores were calculated.  • Appendix N contains more details about the exact simulator performance under different conditions. Our code is publicly available, and links to the code and data can be found at  https://generallyintelligent.com/avalon .

## A.1 Terrain, biomes, scenery, objects

The base terrain is randomly generated via an iterative process of subdivision, bicubic interpolation, and adding various types of random noise at progressively smaller scales (see the  build_outdoor_world_map  function in the code for an exact specification). This heuristic approach is designed to mimic the rugged surfaces of natural landscapes and mountains without using a significant amount of compute. A plane of water intersects the island at a specified height to form the surrounding body of water as well as pools within. The terrain slope, elevation, and distance from water map every point to a set of 7 visually distinct biomes:  tropical rain forest ,  tropical seasonal forest ,  temperate rain forest ,  temperate deciduous forest ,  grassland ,  temperate desert , and  subtropical desert . Additional logic shapes regions around water into more specialized biomes such as beaches and swamps ( water ,  fresh water ,  coastal ,  swamp ,  dark shore ) and adds a few bigger mountains and hills (whose surfaces are either  bare , in which case they can be climbed, or  unclimbable , in which case they cannot) to the landscape. We have tuned the resulting distributions to get worlds that feel both natural and high-variance. The island is then populated with foliage in a biome-dependent manner, with different levels of noise on the borders and density variations that create clearings for the player or agent to walk through. Scenery elements include four types of trees, a bush, a flower, and a mushroom. Their arrangement and density are biome-specific — for example, tropical biomes have mostly palm trees. Each piece of scenery populates specific biomes at specific spatial densities, with additional logic at biome boundaries and near coasts. All scenery objects have noise applied to their colors, scales, and orientations. Trees are climbable so that players can use them to escape predators or gain a better vantage point, while other smaller scenery items are non-colliding (agents can simply walk through them). In addition to scenery objects, a series of interactable items are scattered about in different locations and densities depending on the task. These include logs, sticks, large heavy boulders, medium-sized stackable stones, and fist-sized rocks. Many tasks require that these items be either maneuvered into new positions or used as tools or weapons. Figure 3:  Scenery objects.  Left to right: maple trees, acacia trees, fir trees, palm trees, and a composite shot that includes bushes, flowers, and a mushroom. Figure 4:  Interactable items.  Left to right: log, boulders, sticks, and rocks.

## A.2 Obstacles

To procedurally enforce obstacles between the player and food, the terrain is shaped into irregular rings encircling either the player spawn point or the food location. These rings can consist of cliffs, chasms, ridges, or sudden drops that make it impossible for the player to get to the food without accomplishing the relevant task. Compositional tasks use concentric rings to impose a series of obstacles between the player and food. Noise is added to these rings to make them resemble more natural formations.

## A.3 Fruit trees and fruits

For most individual tasks, the available food item is a fruit that can be found either on or under a large fruit tree. Fruit trees are taller and more visually striking than other foliage (see Figure 5) so that identifying distant food is not akin to spotting a needle in a haystack. The canonical fruit is an apple, but there are nine others that differ from it in some key aspect. These were selected not only for visual variety but to challenge the agent in different ways; for example, some require special preparation to become edible, while others are delicate and can become inedible with the wrong treatment. The full list of fruits and their properties is provided in Table 2 (the word “fruit” is used loosely here, as it also includes honeycombs and avocados). There are also carrots, which are the only food that can grow anywhere i.e. not restricted to be found under a tree or in a building. The only tasks that feature a fruit other than the canonical apple are  eat , which has a single fruit that can be any of the 10, and the compositional task  survive , which has multiple different types of fruits scattered around the island.

## A.4 Buildings and doors

In addition to the natural features of the island, Avalon also includes procedurally-generated buildings which can be entered and explored. Buildings are used as an alternative site for placing fruits and can serve as a site for some tasks such as  open ,  push ,  navigate  and  stack . Buildings may have one or more stories and parts of the buildings may be climbable. They are also guaranteed to contain at least Figure 5:  Fruit tree and fruits.  Top left to bottom right: fruit tree, banana, cherry, apple, mulberry, fig, coconut (left: on tree, right: opened), orange (left: on tree, right: opened), avocado (left: on tree, right: opened), honeycomb (left: on tree, right: dirty), carrots. Table 2: Fruits and their properties. one piece of food. The  open  task takes place exclusively in buildings, with a range of door types and locks corresponding to different difficulties of the task. Doors can be unlocked or locked, and rotating (so the agent can push through it), or sliding (so the agent must drag it to the side). Doors can have between zero and three locks. There are three types of locks: a deadbolt that must be slid to the side, a rotating bolt that must be rotated to the side, and a timed button switch that toggles the ability to move the locks or open the door.

## A.5 Predators and prey

There are 17 animals (nine predators, eight prey) that appear in certain tasks. Animals were selected to span a wide range of behaviors and game mechanics, described in detail below. Depending on the task, animals are either populated randomly or clustered near the food. The full list of animals and their properties is provided in Tables 3 and 4 and screenshots are provided in Figures 6 and 7. In the remainder of this section, we briefly describe the axes of variation. Table 3: Predators (ordered from easiest to hardest) and their properties. Table 4: Prey (ordered from easiest to hardest) and their properties. Domain.  Animals are either confined to the ground or exclusively fly. Some ground animals can climb. All animals can enter buildings except bears and hippos, which are too big to fit through doors. Activation and deactivation.  At any given moment, an animal is either  active  or  inactive , with inactivity the default. Different predators and prey have different activation (deactivation) conditions for when they start (stop) chasing or fleeing the player, respectively. For example, some only activate when the player is nearly on top of them while others activate from a wider radius. Activation conditions are often mirrored in deactivation conditions, though this is not always the case; for example, hawks and wolves both activate when the player enters their territory, but wolves continue the pursuit even after the player exits their territory. Most predators and all prey can be deactivated by killing, but two predators are indestructible (hippo, bear). Another way to deactivate some predators is by triggering the Outside Domain condition. For example, predators that can neither fly nor climb can be deactivated by climbing a tree. Hippos and bears are too big to fit through doors so going inside a building triggers this condition for them. Some special activation conditions: "Must See You" means the player must be within the predator's field of view and "You Don't See It" means it sneakily attacks when the player is not looking. For deactivation conditions, "You See It" is the mirror of "You Don't See It" and "You Stop Moving" is akin to the freeze defense strategy for not getting eaten. Idle behavior.  While inactive, animals display different default idle behaviors. Some move randomly (fast or slow) while others move along fixed periodic trajectories. Some only prowl within their territory. Static Avoidant animals are static until the player approaches, then slowly move away to increase their distance from the player, then finally activate if the player get within their activation threshold. Active speed, attack and defense stats.  When active, some animals are slower than the player, while others are faster. Some predators attack repeatedly (Persistent) while others attack once and then leave for a bit (Temporary Respite). Bees attack once and then die (Permanent Respite). Attacks can be high damage or low damage. A single snake attack is fatal. Some animals can be disabled with one hit, others need multiple hits, and some are indestructible. Figure 6:  Predators.  Top left to bottom right: wolf, bear, jaguar, crocodile, hippo, snake, hawk, eagle, bee. Figure 7:  Prey.  Top left to bottom right: turtle, squirrel, rabbit, deer, mouse, frog, pigeon, crow.

## B Reward function

In all tasks, the agent's reward is given by the same reward function of its energy level over time. Agents begin each episode with a certain initial energy and must inevitably expend energy in moving to complete the task. The only way to gain energy is by eating. In addition to energy expenditure from motion, energy can be lost by taking damage from enemies or falls. Episodes terminate for one of three reasons: i) The agent energy goes to zero. 
 ii) The agent eats all the food in the episode (the episode ends 10 frames later). 
 iii) The episode times out. The determination of the initial energy, definition of energy expenditure, and implementation of termination conditions vary slightly between training and evaluation settings, described in Section B.1. The formulas for each energy term and values of associated constants are provided in Section B.2.

## B.1 Training and evaluation

During training, the agent receives dense reward equal to its framewise change in energy level, together with framewise penalties. This reward  R  is the sum of any energy gained from eating  E_{\text{food}}  minus any energy lost from damage due to falling  E_{\text{fall}}  and predators  E_{\text{predators}} , minus the movement penalty  P_{\text{move}}  and frame penalty  P_{\text{frame}} , Each energy term is discussed in detail in Section B.2. During training and evaluation, the initial energy level  E_0 = 1 . The time limit during training depends on the difficulty level and task, tuned such that 98% of humans trials lie within the time limit. These time limits improve compute efficiency during training, terminating episodes in which the agent cannot perform the task.

## B.2 Energy terms

Each of the energy terms in Equation 1 are described below, with all constants listed in Table 5. Food.  Food is eaten once it is held and within 0.5m of the head. All food (fruits and prey) provides  E_{\text{food}} = 1  upon being eaten, except for bananas which provide  E_{\text{food}} = 0.5  (see Table 2). Falling.  Fall damage occurs if the vertical speed of the agent  v_{\text{vertical}}  is above a certain threshold  v_{\text{threshold}}  when colliding with the ground, with the energy penalty from a fall given by The energy coefficient  C_{\text{fall}}  and threshold  v_{\text{threshold}}  are set so the agent of body mass  m_{\text{body}}  would not take damage from any fall smaller than 12 meters and take lethal damage when falling from anything greater than 28 meters. See Table 5 for details. Predators.  The energy cost of predator attacks depends on the predator as specified in Table 3.

## B.3 Penalty terms

The penalty terms differ from the energy terms in that they do not modify the total agent energy: they do not modify the ability of the agent to survive or act in its environment. Rather, these terms are designed to assist the agent in choosing more appropriate actions which could improve learning and generalization. As such, they contain a number of parameters that may be tuned to optimize agent performance. Frame.  Our scoring criteria (Section M) adds a bonus if all the food on a level is consumed before the end of the episode. This bonus is proportional to the number of frames remaining before the timer would terminate the episode. We use the same proportionality constant used for scoring ( 1e-4 ) as our frame penalty  P_{\text{frame}} . Movement.  As stated in Section 3, the agent controls its head, body, and two hands. The head and body are generally controlled simultaneously. Translating the head in any direction except vertically translates the entire agent. Similarly, rotating the head about the vertical axis (i.e. looking from side to side) reorients the entire agent. However, when it comes to the vertical axis, the agent’s head and body are treated as distinct. When translating the head vertically, only the head moves; this is to enable crouching, during which the agent is still resting on the ground but its vantage point is lower. To vertically translate the agent’s body as well, such that its distance from the ground changes, the agent must jump, climb, or fall. Similarly, rotating the head about axes other than the vertical axis only reorients the head, not the body. This is so that the agent can look up and down or tilt its head to look at things without doing somersaults. With this formalism in place, the energy cost of movement can be stated as the sum of kinetic  \Delta K  and gravitational potential  \Delta P  energy changes for each of the agent’s head, body, and two hands, where the energy costs of the body capture all global translations and rotations of the full agent and the energy costs of the head and hands are calculated relative to the body. For the head, there is only a potential energy cost to penalize keeping the head tilted; there is no kinetic energy cost for crouching. The overall movement coefficient  C_{\text{move}}  and the kinetic and potential energy terms are determined by the physics-based formulas, where  C_{K_i}  and  C_{P_i}  are coefficients that can be tuned,  \bar{\mathbf{v}}  is the average velocity over the last step,  \Delta \mathbf{v}  is the change in velocity,  m_i  is the mass of the part  i ,  g  is the gravitational acceleration, and  \Delta h  is the change in height over the last step. The constants used to calculate these energy terms are listed in Table 5. During training,  C_{\text{move}}  was tuned with other hyperparameters. The body part masses were set to be close to the average human equivalents. Table 5: Parameters used to calculate fall and movement energy values.  C_{\text{move}}  is a hyperparameter that can be adjusted for training. During evaluation,  C_{\text{move}} = 0

## C Observation space

The agent’s observation space is made up of  96 \times 96 \times 4  egocentric RGB + depth images and the following 23 proprioceptive inputs: •  4 \times 2  variables (3D vectors) corresponding to the framewise change in position and rotation of the agent’s body, head, and two hands. 
 •  3 \times 2  variables (3D vectors) corresponding to the position and rotation of the agent’s head and two hands relative to its body. •  2 \times 1  variables (booleans) corresponding to whether each hand is colliding with anything. 
 •  2 \times 1  variables (booleans) corresponding to whether each hand is grasping something. 
 • 4 variables (floats) corresponding to the agent’s energy expenditure from movement, energy lost from enemy attacks, energy gained from eating, and current energy level. 
 • 1 variable (float) corresponding to the frames remaining before the episode is terminated by the timer.

## D Action space

The full 21-dimensional action space includes  3 \times (3 + 3)  continuous degrees of freedom for the 3D translation and rotation of the agent’s three components (head and two hands), as well as  2 \times 1  binary grasp actions for the hands and 1 binary jump action. This action space roughly corresponds to the controls of a VR headset and was used for all experiments in the paper. Human players were also allowed to use the controller joystick for movement and rotation, but these inputs were carefully mixed into the simpler action space in which the agent acts so that the two could be exactly equivalent. We also provide a reduced 9-dimensional action space which maps to mouse and keyboard controls. In this reduced action space, there is a single translational degree of freedom for forward/backward motion, with the agent always moving in the direction it is facing, which is controlled by two angular degrees of freedom (pitch and yaw). In addition to the  2 \times 1  binary grasp actions and 1 continuous jump action, this setting also has 1 discrete action for eat and  2 \times 1  discrete actions for throw.

## E Factors of variation

The procedure that generates Avalon’s levels is parameterized by many  factors of variation  which determine the structure and appearance of the generated world. These factors of variation can be broken down into settings that affect terrain (shape and color), scenery (trees, bushes, flowers, etc), environment (sky, lighting and graphics options), buildings, items (animals, tools, and food), and tasks (distributions over factors like how far to jump, how many enemies to include, etc). Two levels generated from the same factors of variation will share these high-level features but will differ in various low-level details like object positions and the exact topography of the island. Due to the focus on variety and diversity in Avalon, the factors of variation are mostly defined in code rather than in external configuration files. This also allows for better specification of allowed values (via type signatures).

## E.1 Terrain

The terrain (base world geometry) is generated via repeated subdivision and addition of various types of noise. The  WorldConfig  object defines 34 properties that control this procedure, including  fractal_iteration_count ,  noise_scale_decay , and  size_in_meters . See the code for a complete list and documentation for each value. The  generate_world_config  function gives an example of how to dynamically generate interesting, varied  WorldConfigs . The  build_outdoor_world_map  converts a  WorldConfig  into a  HeightMap . It can easily be swapped out for any other approach to generating a  HeightMap  (grid of heights). After generating the base terrain, the world is broken into different “biomes” that control which scenery objects will be placed in that region, as well as the colors used for the terrain in that region. Some of these biomes affect the height of the world (ex: the beach biome controls the erosion of shores near the ocean). This entire process of biome assignment and calculation is controlled by the  generate_biome_config  function, which generates a  BiomeConfig . This object has 43 properties such as  beach_slope_cutoff ,  swamp_elevation_max , and  rock_color_noise . See the code for a fully documented list.

## E.2 Scenery

In order to strike a good balance between visual complexity and performance, biomes in Avalon are populated with instanced scenery models. These models are given per-instance and per-vertex color noise, as well as per instance scaling and rotation variation without really incurring any significant performance overhead. These attributes are controlled by the  FloraConfig  and  SceneryConfig  objects, which together define 10 attributes like  density ,  border_mode , and  correlated_scale_range .

## E.3 Environment

All aspects of the Godot  WorldEnvironment  and  Sky  objects are procedurally generated, as well as all aspects of the Sun light that is created in each scene. This enables varying factors such as  sun_latitude ,  fog_color  and  tonemap_mode . We give examples of setting 46 of these, though all supported properties of these objects in Godot can be set directly. See the Godot documentation for more details on each setting.

## E.4 Buildings

Buildings are used both as components of compositional worlds and as variants of the basic tasks (ex:  explore  tasks are set either inside of a building or in an outdoor, natural world). The appearance of buildings can be changed via the  BuildingAestheticsConfig , which contains 20 attributes like  desired_story_count ,  window_width  and  trim_color

## E.5 Items

All items, including animals, have a variety of attributes which can be set directly and differ per-item. See  items.py  for a complete definition of all attributes. Each item also has  safe_scale  and  base_color  attributes, which can be used to explicitly set the size and color of each object. Animals each have a variety of behavior-dependent attributes like  speed ,  activation radius , etc that can be altered for each instance.

## E.6 Tasks

Tasks are each generated by a single function, parameterized only by difficulty. This function converts that difficulty value into all other lower-level factors of variation (e.g.  jump  distance for  jump , path width for  move , etc). See the corresponding  .py  files in the  tasks  folder for a complete specification of each task's parameters. Tasks which can be composed also contain a single function to create the given type of obstacle (ex: a gap for jumping over, a path for walking along, etc), and these functions are used by  compositional.py  to implement the compositional tasks.

## F Task definitions

Each task in Avalon maps to a set of generated worlds. Each world is set up such that the agent must usually complete the task in order to successfully reach the food. Each task has a variety of individual parameters, controlled in aggregate by the difficulty parameter, which ranges from 0 to 1. See Figure 9 for examples of how tasks vary by difficulty. Almost every task world has only one fruit or prey; after the agent eats the fruit or prey, the episode ends. The two exceptions are  gather , which has multiple fruits, and  survive , which has multiple prey and fruits. To enable visibility for the agent in rugged terrain, the fruit in each world is always on a fruit tree or inside a building. Each task is set up with the goal of isolating the skill being learned. Thus, all tasks only have apples as canonical fruit at all difficulty levels, with the exception of more difficult  eat  worlds, which contain harder-to-eat fruits;  hunt  and  throw , which contain only prey and no fruit; and  survive , which can contain all fruit and prey types. This is intended to isolate the skills needed for each task from the skills needed to eat more complex fruits. Additionally, in most tasks, with the exception of  explore ,  find ,  gather , and  survive , the agent and the fruit spawn in locations such that the agent can see where the fruit is at spawn. In this case, the agent may not be facing the fruit at spawn, but upon turning in place will be able to see the fruit. This is intended to isolate the skills required for each task from the skills required for exploring the world, which is tested in isolation in  explore , as well as in the compositional tasks.

## F.1 Basic tasks

There are 16 "basic" tasks. Each basic task generates a world in which the agent usually must complete that task in order to reach and eat the food. Figure 8:  Basic tasks.  All 16 basic tasks. For each task, the world is generated such that the agent must complete the task in order reach and eat the food. Eat.  In  eat , the agent starts in front of the fruit, and must grab the fruit and bring the fruit to within a radius of its head in order to succeed. As difficulty increases, worlds contain different types of fruit that must be opened in more complex ways (see Appendix A for details on fruit types). Move.  In  move , the agent starts a distance away from the fruit, and must move its body to reach the fruit. As difficulty increases, the agent spawns farther away from the fruit, and the paths leading to the fruit become narrower. At the highest difficulties, the agent must traverse thin, multi-segment paths across chasms in order to reach the fruit. Jump.  In  jump , the agent must jump over a chasm to reach the fruit. As difficulty increases, the chasm gets wider, and the area that can be jumped across gets narrower, such that the agent must figure out where it can successfully land a jump. If agents fall into the chasm while jumping, they can climb back out to the original side and retry the jump. Climb.  In  climb , the agent must climb up a cliff wall in order to reach the fruit. The climbing motion requires repeatedly bringing one hand up, grabbing the cliff, pulling the hand downward, and then doing the same with the other hand, without letting go of the cliff. As difficulty increases, the cliff gets taller so the agent must climb further, and the climbable path gets narrower so the agent has less area to grab onto. The climbable path also contains multiple, angled segments at higher difficulties, requiring more than just climbing in a single straight line. Scramble.  In  scramble , the agent must combine walking, jumping and climbing to move over terrain to reach the fruit. As difficulty increases, the terrain becomes more mountainous, requiring more climbing and less jumping. Descend.  In  descend , the agent must descend down a cliff. This can be accomplished by climbing down, or falling while grabbing the cliff wall periodically. On easier difficulties, there is also a platform partway down that can be used to safely descend by first falling on the platform, then falling the rest of the way. If the agent just jumps off the cliff randomly (without falling on this platform), it will take damage. As difficulty increases, the cliff gets taller so the agent will take more damage (and, at the highest difficulties, die). The path that can be grabbed also gets narrower, so the agent must figure out where to descend from. Throw.  In  throw , the agent must throw a rock at prey in order to kill it, and then eat it in the same way fruit is eaten.  throw  worlds only have prey, and no fruit. As difficulty increases, the agent starts farther away from the prey, and the world gets larger, so the prey can run away. Additionally, the types of prey that spawn are harder to hit (e.g. pigeons that fly, instead of frogs that sit on the ground and hop slowly). Hunt.  In  hunt , the agent must find the prey and kill it either by throwing a rock, or hitting it with a stick. As difficulty increases, the world gets larger, the types of prey that spawn are harder to hit, and fewer tools spawn for hunting. The agent may only get one rock, or one stick. Fight.  In  fight , the agent must use sticks or stones to fight a predator that is guarding the food in order to reach and eat the food. While it is not strictly necessary to defeat the predators in order to reach the food in some levels, as difficulty increases, the predator types that spawn are more aggressive and more difficult to hit, and there are more of them, effectively forcing the agent to fight. Avoid.  In  avoid , the agent must avoid predators that are near the food in order to reach and eat it. Unlike in  fight , the agent is not provided with sticks or stones for fighting the predator. As difficulty increases, the number of predators increase, and more aggressive types of predators become more likely to spawn. Push.  In  push , the agent must push a heavy boulder into a position where it can be used as a stepping stone for jumping onto the cliff ledge where the fruit is. As difficulty increases, the boulder gets heavier and more difficult to control, and must be pushed farther. This task has both indoor and outdoor variants. Stack.  In  stack , the agent must stack stones and jump on top of them in order to reach the cliff ledge where the fruit is. At low difficulties, the agent only needs to use one stone to reach the top. As difficulty increases, the cliff ledge gets taller, so more layers of stones need to be stacked to reach the top. At the highest difficulty, the agent must stack a pyramid with four layers of stones in order to reach the top. This task has both indoor and outdoor variants. Bridge.  In  bridge , the agent must pick up a log and lay it across a chasm in order to create a bridge to walk over the chasm. At low difficulties, the log is sometimes already in the solved position. As difficulty increases, the log starts out farther away, and the width of the chasm section that is narrow enough to be bridged shrinks, so the agent must find the right place to bridge the chasm. Open.  In  open , the agent starts inside a building and must open a door to get into the room where the fruit is. See Appendix A for details on types of doors and locks. At the easiest levels, doors are rotating and can be walked through. As difficulty increases, more difficult variants of doors (such as sliding doors and doors that must be pulled rather than pushed) with more difficult locks (such as the timed switch) appear more often. At the highest difficulty, doors can all three locks. Carry.   carry  spawns a task world that requires objects from one of  throw ,  fight ,  stack , or  bridge , and then moves the objects a distance away from where they would normally spawn. Thus, the agent must carry the object to the location where it will be used. As difficulty increases, objects spawn farther away and must be found and carried a longer distance. Explore.  In all other basic tasks, fruit or prey is visible from where the agent spawns. In  explore , fruit cannot be seen from where the agent spawns. Thus, the agent must explore the terrain in order to find the fruit. This task has both indoor and outdoor variants. In the outdoor variant, as difficulty increases, the world gets larger and the terrain rockier, making it more difficult to spot and reach the fruit. In the indoor variant, the building to be explored becomes progressively larger with higher difficulties.

## F.2 Compositional tasks

There are four "compositional" tasks. Compositional tasks generate worlds in which the agent must complete a sequence of multiple basic tasks in order to get food. To prevent the agent from bypassing any task, compositional tasks use concentric rings of terrain obstacles to impose a sequence of task obstacles, or spawn buildings that require one of the basic tasks. Navigate.  In  navigate , the fruit is visible from where the agent spawns, and the agent must navigate through a variety of basic tasks to reach the fruit. As difficulty increases, the world gets larger, with more difficult terrain, and the number and difficulty level of basic tasks that must be solved also increases (up to a maximum of four basic tasks). Find.   find  is like  navigate , but the fruit is not visible from where the agent spawns. In order to achieve maximal score on the task, the agent must find the fruit and solve a variety of basic tasks to reach the fruit before time runs out. As difficulty increases, the world gets larger and more mountainous, and the number and difficulty of basic tasks to be solved along the way also increases. Gather.   gather  is like  find , but with multiple pieces of fruit in the world. None of the fruit is guaranteed to be visible from where the agent spawns. Thus, the agent must find and reach all pieces of fruit before time runs out, solving obstacles along the way, in order to achieve max score on the task. As difficulty increases, the world gets larger and more mountainous, the distance between fruits increases, and the number and difficulty of basic tasks to be solved increases. Survive.   survive  is like  gather , but with prey and predators scattered about the world, in addition to fruit and some basic task obstacles. Unlike in  navigate ,  find , and  gather , obstacles don't necessarily appear at all or need to be solved in sequence. Instead, the agent's goal is to survive as long as possible by eating prey or fruit, avoiding predators, and solving obstacles in order to find more food. As difficulty increases, the world gets larger and more mountainous, and there are more predators of higher difficulty and fewer prey and fruit.  survive  at the high difficulty levels is in some ways the pinnacle task: the agent usually needs to have learned a wide range of skills from all other tasks in order to achieve maximum scores on the most difficult survive levels.

## G.1 World generation

A fixed set of 50 worlds for each task was generated to evaluate human and agent performance. Each world was generated with a unique seed while difficulties ranged between 0.0 and 1.0 and the exact distribution of difficulties was task dependent. For each task type, 20% of the worlds were set to Figure 9:  Task variation with difficulty.  Examples of how tasks vary as difficulty increases.  stack  0.0 is already solved and the agent just needs to jump on the blocks to reach the landing;  stack  1.0 requires the agent to create a pyramid of blocks that is three blocks high in order to reach the landing.  climb  1.0 requires climbing a higher cliff, on a narrower path than  climb  0.0. In  open  0.0 the agent can directly walk through the door;  open  1.0 requires the agent to unlock three locks, including a timed switch, to open the door.  move  0.0 requires moving a short distance on flat land to get food, whereas  move  1.0 requires moving a longer distance on a narrow path surrounded by cliffs. have difficulty 1.0 and the remaining worlds had difficulties that were evenly spaced between 0.0 and 1.0. For  hunt ,  avoid  and  eat , a world for each type of food, prey and predator were generated to guarantee that each type of entity was present in an evaluation world. For  avoid  and  hunt , the difficulties of these forced worlds were set to 1.0, while for  eat  the difficulty was set to 0.5. The forced levels were created first and then the same procedure as the other tasks was used to generate the remaining worlds. Figure 10:  Compositional tasks.  All four compositional tasks. Compositional tasks generate worlds in which the agent must complete a sequence of multiple basic tasks in order to get food.

## G.2 Replaced worlds

During data collection, participants reported worlds that were impossible or very difficult to complete. Of the original 1,000 levels generated (50 levels  \times  20 tasks), there were a total of 30 levels (3%) that had any issues at all. Ultimately, 10 levels (1%) were replaced. Four levels were impacted by 2 bugs which have since been fixed. When these issues were encountered during data collection, we simply generated new worlds with an increased seed (and the same difficulty level) to replace them in the evaluation set. Twenty-six levels did not have successful playthroughs in the initial round of data collection. These levels tended to be on the highest difficulty setting and most difficult tasks, and are listed in Table 6. Of those 26 levels, we were able to eventually solve 20 by simply having more players try them. Of those 20 levels, 9 required multiple attempts from the same individual before they were ultimately solved; however, multiple attempts were needed only due to the difficulty of execution, not because they required advance knowledge of the level, and thus it seems likely to us that the level could have been solved on the first attempt given enough skilled players (but unfortunately we did not have enough study participants to verify). The remaining 6 did not get successful playthroughs even after additional attempts, and thus were replaced. However, only 1 of these 6 was truly impossible (the terrain is extremely jagged, leading to a situation where the fruit spawned in an area that is unreachable) whereas the others were merely very difficult. In summary, 10 of the generated worlds (1%) were replaced, but of those only 1 (0.1%) was due to a level actually being impossible. We reiterate that the evaluation set does not contain any unsolvable worlds as they were all replaced, but we report these issues here for the sake of transparency.

## G.3 Time Limit

To limit the amount of time a participant could spend on a world, the following time limits were used: • 15 minutes for  navigate  and  find  
 • 10 minutes for  survive ,  gather ,  stack ,  carry  and  explore  
 • 5 minutes for all remaining tasks When evaluating the agent, the maximum roll-out length was limited to be the same as human participants. See Appendix M for more information on evaluation and scoring.

## H.1 Selection and compensation

Thirty participants were drawn from a pool of volunteers who indicated interest and an ability to commit 10 hours to the study during a week-long time-frame. Participants were asked to sign a participant consent form that included information regarding the purpose, procedure, risks and discomforts, potential benefits, costs, payment, confidentiality, and the subject’s rights during and after the study (see AdultConsentForm in the supplemental materials). Table 6: List of 30 generated evaluation worlds for which there were initially no successful playthroughs. After inspection and additional playthroughs, 10 worlds were replaced in the evaluation set (see comments column), although only 1 was actually unsolvable after fixing bugs. Risks and discomforts included cybersickness and short-term effects following VR use. The participants were also warned that there is a small chance that their identities could be inferred from their anonymized motion data. Participants were given an option to either be thanked or to remain entirely anonymous. All personally identifying information (i.e. names) has been removed from the dataset of human motion data (replaced with a random unique identifier). Participants received payment in one of two forms: 1. Any participant who did not already own an Oculus Quest 2 headset received a new one along with a supplemental battery pack, and any hours spent over the required 10 hours were reimbursed at $30/hour. 
 2. Any participant who already had an Oculus headset received a supplemental battery pack and was reimbursed at $30/hour for the full 10 hours and any time they spent over that. The rate of $30/hour was set to be fair to all participants regardless of whether they had an Oculus headset to start: an Oculus Quest 2 headset costs $300, which is commensurate with a $30/hour reimbursement over a 10-hour study period. Participants were also offered prizes of $200, $100, and $50 for the top three performers to incentivize focus and maximum effort during the evaluation worlds, since the gameplay can become somewhat monotonous. These amounts were selected as a balance between incentivizing focus while avoiding causing any undue stress to participants. If a participant didn't finish the required 10 hours, they were given the option to send back their headset (if applicable) and get reimbursed for the time spent or pay us back for the difference in hours. Figure 11:  Left:  Distribution of aggregated scores for each player.  Right:  Distribution of number of episodes played. Two users completed almost every evaluation level, while most completed around 200 levels. In total, approximately $12,000 was spent on participant compensation between Oculus headsets, supplemental battery packs, and cash prizes/reimbursements.

## H.2 Instruction and feedback

For the first 3 hours of the study, participants were asked to set up their Oculus headsets by installing our Avalon APK. They received a document with setup instructions as well as information about basic game mechanics such as the game controls and the tools and animals they would encounter in the environment (also included in the supplemental materials, see “Oculus Setup Practice Instructions”). Participants were then asked to complete at least 2 practice worlds for each of the 20 tasks. This data was not included in the evaluation set, and is not included in the recorded human data. For the next 7 hours of the study, participants were asked to complete a series of evaluation worlds chosen at random from the pool of 1,000 evaluation worlds. No participant ever saw the same level twice. Two participants played almost 1,000 levels, and most players completed around 200 levels (Figure 11), for a total of 6,145 played episodes. Participant motion data was captured during these evaluation worlds. Participants were allowed to “reset” if they ended up failing or getting stuck in a world. These resets were not counted towards the 5 playthroughs per world. This ability to reset was added to reduce stress on the human participants (whether due to perceived failure or real-life interruptions). In a feedback survey after the study, participants reported an average overall satisfaction with the study of 4.57 out of 5 and 30 out of 32 of the participants said they would like to be considered for future studies.

## H.3 Analysis of performance

Most levels were quite simple, leading to final episode scores clustered tightly around one (Figure 12). Aggregated player performance was quite similar as well (Figure 11), with low performing users playing on just a few high difficulty levels. In 962 episodes, the player either died during the episode or reached the time limit without eating food, getting a final score of zero. The remaining 5,183 episodes had a score greater than zero, a condition we refer to as a “success.” The distribution of these successes across tasks is shown in Figure 13. Human performance is fairly consistent across most tasks, with some of the compositional tasks presenting more of a challenge. Figure 12: Distribution of scores for each episode. Most players were able to eat all the available food in about the same time for most episodes, leading to a narrow distribution of scores. Figure 13: Success rate of players on each task. Success is defined as completing an episode with a score greater than 0.

## I Curriculum

All task generators are parameterized, at the highest level, by the task id and the “difficulty”  d_t  (a float in the range  [0, 1] ). Each environment generation worker process maintains a mapping from task id to the current maximum difficulty  d_t  (all initialized at 0), and generates a level from a uniform distribution  U[0, d_t] . When an agent succeeds (or fails) at a generated environment, the  d_t  is increased (or decreased) by  H_t , a hyper-parameter that controls how quickly the task curriculum adjusts. An agent succeeds at its environment if it eats all of the available food before the episode timer is finished.

## J Hyperparameters

See table 7 for hyperparameters used for training PPO, IMPALA, and Dreamer. Table 7: Hyperparameters used for training.

## K Training compute

Our training took place in containers, with 8 containers to a machine. One GPU was assigned per container. The whole machine provides 8x Nvidia GeForce RTX 3090 (with 24GB RAM), and is powered by 2x AMD EPYC 7313 CPUs (with 16 cores/32 threads each) with 256GB RAM total. Training time was approximately 30 hours for a single run.

## L Results for other training runs

Table 8 shows the optimality gap scores corresponding to our main experimental runs (whereas Table 1 reports average scores). The same trends are apparent in both tables: the compositional tasks are much more difficult than most of the basic tasks, and the gap between human scores and even the best-performing networks are emphasized here. Tables 9 and 10 show the average scores and optimality gap scores respectively for IMPALA under each of the four training conditions, averaged over three training runs with different seeds. A few salient conclusions may be drawn from this data. First, comparing MT-TB with the single task baselines ST-B, one can see that the similarity of environments has enabled significant transfer learning. Several tasks like  jump  and  climb  are too challenging to learn on their own, but may be achieved by training on other tasks. While the performance of basic tasks such  eat  and  move  somewhat lower in the multi-task setting than the single task setting, the difference is small enough that it is more likely due to fewer exposures to each task in the multi task case, rather than being a case of catastrophic forgetting. Second, we note that training on only the basic tasks seems to be the most effective use of training time. While an agent training on all tasks or even just the compositional tasks does learn a broad variety of basic skills (for example, despite never seeing these levels, MT-TC has significant performance on  eat  and  open ), these agents do not outperform an agent trained on just the basic tasks when measured on either the basic or compositional aggregate scores. We suspect that these tasks may be too difficult for the agent to make much progress at this stage in training. In Figure 14, we show a performance profile by task for each network (IMPALA and PPO) on the MT-TB training setting (for which results are shown in Table 1 in the main paper). For many tasks, humans had quite consistent performance, due to the simplicity of the tasks and sparse reward. This leads to the relatively flat horizontal sections of the graph extending from 0 to 1, where the y-axis indicates the fraction of these levels in which the agent got the single available food. For many tasks Table 8: Optimality gap results for Table 1. Lower scores are better. the scores drop off as they approach  \tau = 1 , as the tasks get harder and there is some more nuance to performance beyond the binary "found food or not" (e.g., from tasks that take longer and or where there is some risk to humans of dying). From the tasks with any amount of super-human performance, these seem likely to be cases where the episode was extremely short and the agent moved faster and oriented itself more quickly than the average human. From the graphs in Figure 14, we can also see that most of the advantage for IMPALA over PPO comes from better performance on the basic tasks. Both networks perform quite poorly on the compositional tasks, mostly only succeeding at  survive , which often has plentiful food, making nonzero scores easier to achieve. One can get a sense for the relative difficulties of the tasks. Some, like  open  and  avoid , seem surprisingly easy, though this is likely due to relatively forgiving levels at low difficulties rather than due to highly effective agents. We visualize the generalization performance of single task training in Figure 15. One can see that training on other tasks readily transfers to  eat  or  move , but that the agents are not able to make much progress on either the trained task or the other tasks. We finally show learning curves for IMPALA using MT-TB in Figure 16. One can see that performance on some tasks continues to improve throughout training, and so the agent may continue to benefit from more steps in the environment.

## M Scoring

For both human players and RL agents, the score of each run is given by Table 9: Mean-human normalized performance for RL agents trained with protocols defined in Section 5.2. MT-TB is trained on the 16 simple tasks, MT-TA is trained on all 20 tasks, and MT-TC is trained on the four compositional tasks. All agents are trained for 50M steps using IMPALA. ST-B results are trained only on a single task for 50M steps, and are evaluated only on that same task. Table 10: Optimality gap results for Table 9. Lower scores are better. Figure 14: Results for RL agents trained on all 16 basic tasks and evaluated on each of Avalon’s 20 tasks (MT-TB). Scores shown are average scores, normalized against mean human performance. Top row and bottom left: scores for IMPALA, PPO and Dreamer on each task. Bottom right: aggregate scores averaged over the 16 basic tasks, the four compositional tasks, and all 20 tasks. where  E_0  is the starting energy level,  E_f  is the final energy level (when the last food is consumed, the agent dies or the episode times out),  T  is the total number of episode frames remaining until timeout, and  P_{\text{frame}}  is a constant penalty per frame. For effective comparison with humans, we do not impose energy penalties for movement during evaluation (i.e. the  C_{\text{move}}  of Appendix B is zero), opting instead for the constant per-frame energy penalty. We use the same  P_{\text{frame}}  during training of  1e-4  when scoring runs and for each task,  E_0  is always set to 1.0. For most tasks, success is simply whether or not the player ate all of the food in the level (and there is usually just a single piece). For tasks which always have multiple pieces of food, namely  gather  and  survive , a run is considered successful if the player ate all food, or ate at least one food and ended the task with more health than they started with. All reported scores are normalized post hoc such that the average human performance on a given task is 1.0 and the performance of a random agent is 0.0.

## N Simulator Performance

One of the major contributions of Avalon is the simulator, which was specifically designed to be high-performance. Table 11 gives the performance for multiple processes on a single 2080Ti GPU, and Table 12 gives the performance for a single process for the same configuration. Each table gives a breakdown of performance, in terms of simulated steps per second, by both the size and complexity of the worlds being simulated, as well as by the graphical options that were enabled. The worlds range in size from Small ( 64\text{m} \times 64\text{m} ) to Huge (almost a square half-kilometer). See Figure 17 for Figure 15: Mean human-normalized performance for IMPALA trained for 50M environment steps on a single task and tested on another task. images of the levels that were used for profiling). The graphical options include which renderer to use (GLES2 vs GLES3) and which options (fancy vs basic). For the "basic" condition, shadows were disabled, lighting was done per-vertex, and all settings were configured for the Godot defaults for mobile rendering. For the "fancy" condition, the opposite was true—lighting was per pixel, shadows were enabled, and all Godot defaults for desktop computers were used. Additionally, GLES3 has MSAA 4x enabled for both conditions, since we use this for our agent because it makes it easier to detect small objects from farther away without incurring much of a performance penalty. Our performance numbers are meant to get as close as possible to the performance numbers reported for Habitat. We used a 2080Ti GPU in order to make our numbers more directly comparable. We also show results on a 3090 GPU (which was used for all of our other experiments) in Tables 13 and 14. The 2080Ti was paired with a Intel i9-10980XE CPU, while the 3090 had an AMD Ryzen 9 5950X. We also rendered at 128 x 128, a higher resolution than used in the rest of our paper, in order to be more directly comparable. Like Habitat, our benchmarking setting consisted of reading actions from a pipe and writing outputs to a file (no networks were being trained or run on the same GPU). Unlike Habitat, our simulator works in a straightforward loop of reading actions, stepping physics and gameplay logic, and rendering the resulting frame, while their fastest numbers came from an interleaved setting that delayed observation of the effect of an action by an extra frame. We also do not require any frame pointer passing or other complex integrations, making it easier to integrate our simulator into any network training setup, including distributed settings. It should be noted that this performance is close to the maximum possible, given the hardware and software available today—a non-trivial amount of the time spent in rendering our Small world is spent simply clearing the screen via the glClear call (a necessary component for rendering). The single-process numbers represent the maximum possible speed-up over real-time for a single agent in our simulator (e.g., since we run the simulation at 10 steps per simulated second and a single process Figure 16: Learning curves for mean human normalized performance on each task, using IMPALA and MT-TB for 500m training runs with the curriculum. Each point is the average over 3 different training runs with different seeds. Table 11: Multi-process performance (steps per second) on a single 2080TI GPU on various sizes of worlds (from Small to Huge). on the 3090 can do 4,114 steps per second, we can run an agent at up to 411.4x real time). It should be noted that this speed is not reflective of performance real world–practical training and evaluation is dominated by the network forward and backward passes. The multi-process numbers are more reflective of training throughput and efficiency per-GPU. Table 12: Single-process performance (steps per second) on a single 2080TI GPU on various sizes of worlds (from Small to Huge). Table 13: Multi-process performance (steps per second) on a single 3090 GPU on various sizes of worlds (from Small to Huge). Table 14: Single-process performance (steps per second) on a single 3090 GPU on various sizes of worlds (from Small to Huge). Figure 17: Each of the worlds used for profiling and their sizes, from left to right: Small ( 32\text{m} \times 32\text{m} ), Medium ( 64\text{m} \times 64\text{m} ), Large ( 220\text{m} \times 220\text{m} ), and Huge ( 440\text{m} \times 440\text{m} )

## Figures (text descriptions)

### Figure 1

Figure 11: Two histograms showing player performance and episode completion. The left histogram, 'Player average score distribution', shows the mean normalized level score (0.00 to 2.00) on the x-axis and the number of players (0 to 12) on the y-axis. The distribution is centered around 1.00. The right histogram, 'Distribution of episodes per player', shows the total number of episodes played (0 to 1000) on the x-axis and the number of players (0 to 8) on the y-axis. The distribution is skewed towards lower episode counts, with a peak around 200 episodes.

The figure contains two histograms. The left histogram, titled "Player average score distribution", has an x-axis labeled "Mean normalized level score" ranging from 0.00 to 2.00 with increments of 0.25. The y-axis is labeled "Number of players" ranging from 0 to 12. The distribution shows a sharp peak at a score of 1.00 with 12 players, and smaller peaks at 0.75, 1.25, and 1.50. The right histogram, titled "Distribution of episodes per player", has an x-axis labeled "Total number of episodes played" ranging from 0 to 1000 with increments of 200. The y-axis is labeled "Number of players" ranging from 0 to 8. The distribution shows a peak at 200 episodes with 8 players, and another peak at 1000 episodes with 2 players. Most players completed between 0 and 400 episodes.  Figure 11: Two histograms showing player performance and episode completion. The left histogram, 'Player average score distribution', shows the mean normalized level score (0.00 to 2.00) on the x-axis and the number of players (0 to 12) on the y-axis. The distribution is centered around 1.00. The right histogram, 'Distribution of episodes per player', shows the total number of episodes played (0 to 1000) on the x-axis and the number of players (0 to 8) on the y-axis. The distribution is skewed towards lower episode counts, with a peak around 200 episodes.

### Figure 2

Figure 12: Human episode average score distribution. A histogram showing the distribution of normalized episode scores. The x-axis is 'Normalized episode score' from 0.00 to 2.00. The y-axis is 'Number of episodes' from 0 to 1600. The distribution is highly peaked around 1.00, with a small secondary peak at 0.00.

Data for Figure 12: Human episode average score distribution 
 
 
 Normalized episode score 
 Number of episodes 
 
 
 
  0.00  1000  
  0.25  0  
  0.50  0  
  0.75  10  
  0.80  20  
  0.85  50  
  0.90  100  
  0.95  200  
  1.00  1600  
  1.05  200  
  1.10  100  
  1.15  50  
  1.20  20  
  1.25  10  
  1.50  5  
  1.75  2  
  2.00  0  
 
  Figure 12: Human episode average score distribution. A histogram showing the distribution of normalized episode scores. The x-axis is 'Normalized episode score' from 0.00 to 2.00. The y-axis is 'Number of episodes' from 0 to 1600. The distribution is highly peaked around 1.00, with a small secondary peak at 0.00.

### Figure 3

Figure 13: Success fraction by task. A horizontal bar chart showing the success rate of players on various tasks. The x-axis is 'Fraction of episodes with score &gt; 0' from 0.0 to 1.0. The y-axis lists tasks: eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, throw. The 'eat' task has a 100% success rate.

Data for Figure 13: Success fraction by task 
 
 
 Task 
 Fraction of episodes with score &gt; 0 
 
 
 
  eat  1.00  
  avoid  0.70  
  scramble  1.00  
  move  1.00  
  jump  0.85  
  fight  0.75  
  climb  0.95  
  explore  0.95  
  descend  0.70  
  push  1.00  
  stack  0.98  
  open  1.00  
  carry  0.85  
  bridge  1.00  
  hunt  0.85  
  gather  0.60  
  survive  0.55  
  find  0.55  
  navigate  0.60  
  throw  0.95  
 
  Figure 13: Success fraction by task. A horizontal bar chart showing the success rate of players on various tasks. The x-axis is 'Fraction of episodes with score &gt; 0' from 0.0 to 1.0. The y-axis lists tasks: eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, throw. The 'eat' task has a 100% success rate.

### Figure 4

Figure 14: Results for RL agents trained on all 16 basic tasks and evaluated on each of Avalon's 20 tasks (MT-TB). The figure contains four subplots: IMPALA, PPO, Dreamer, and Aggregate. Each subplot shows the 'Fraction of runs with score ≥ τ' on the y-axis (ranging from 0.0 to 1.0) against the 'Human-normalized score τ' on the x-axis (ranging from 0.0 to 2.0). The IMPALA, PPO, and Dreamer plots show performance for 20 individual tasks, with a legend listing: eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, and throw. The Aggregate plot shows performance averaged over 16 basic tasks, 4 compositional tasks, and all 20 tasks, with a legend listing: IMPALA (basic), IMPALA (compositional), IMPALA (all), PPO (basic), PPO (compositional), PPO (all), Dreamer (basic), Dreamer (compositional), and Dreamer (all).

Figure 14: Results for RL agents trained on all 16 basic tasks and evaluated on each of Avalon's 20 tasks (MT-TB). The figure contains four subplots: IMPALA, PPO, Dreamer, and Aggregate. Each subplot shows the 'Fraction of runs with score ≥ τ' on the y-axis (ranging from 0.0 to 1.0) against the 'Human-normalized score τ' on the x-axis (ranging from 0.0 to 2.0). The IMPALA, PPO, and Dreamer plots show performance for 20 individual tasks, with a legend listing: eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, and throw. The Aggregate plot shows performance averaged over 16 basic tasks, 4 compositional tasks, and all 20 tasks, with a legend listing: IMPALA (basic), IMPALA (compositional), IMPALA (all), PPO (basic), PPO (compositional), PPO (all), Dreamer (basic), Dreamer (compositional), and Dreamer (all).

### Figure 5

Heatmap showing mean human-normalized performance for IMPALA trained for 50M environment steps on a single task and tested on another task. The x-axis and y-axis both list 20 tasks: avoid, bridge, carry, climb, descend, eat, explore, fight, find, gather, hunt, jump, move, navigate, open, push, scramble, stack, survive, throw. The diagonal cells are white, indicating perfect performance on the training task. The highest off-diagonal performance is 0.63 for scramble when trained on stack, and 0.60 for eat when trained on hunt.

Mean performance on each testing task for single task training  
 
 
 Train task \ Test task 
 avoid 
 bridge 
 carry 
 climb 
 descend 
 eat 
 explore 
 fight 
 find 
 gather 
 hunt 
 jump 
 move 
 navigate 
 open 
 push 
 scramble 
 stack 
 survive 
 throw 
 
 
 
 
 avoid 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.02 
 0.00 
 0.00 
 0.00 
 0.00 
 0.02 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 
 
 bridge 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 carry 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 climb 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.03 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 descend 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.07 
 0.02 
 0.01 
 0.00 
 0.00 
 0.01 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.01 
 
 
 eat 
 0.13 
 0.03 
 0.04 
 0.11 
 0.14 
 0.75 
 0.16 
 0.12 
 0.00 
 0.01 
 0.03 
 0.21 
 0.23 
 0.00 
 0.04 
 0.01 
 0.21 
 0.01 
 0.03 
 0.00 
 
 
 explore 
 0.41 
 0.00 
 0.10 
 0.02 
 0.18 
 0.51 
 0.25 
 0.23 
 0.00 
 0.03 
 0.03 
 0.16 
 0.34 
 0.00 
 0.06 
 0.02 
 0.08 
 0.01 
 0.04 
 0.00 
 
 
 fight 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.02 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 find 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 gather 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.02 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 hunt 
 0.24 
 0.00 
 0.04 
 0.01 
 0.19 
 0.60 
 0.20 
 0.16 
 0.00 
 0.02 
 0.25 
 0.13 
 0.24 
 0.00 
 0.07 
 0.04 
 0.08 
 0.03 
 0.08 
 0.00 
 
 
 jump 
 0.14 
 0.01 
 0.05 
 0.05 
 0.07 
 0.33 
 0.06 
 0.12 
 0.00 
 0.00 
 0.06 
 0.11 
 0.13 
 0.00 
 0.01 
 0.05 
 0.11 
 0.02 
 0.02 
 0.00 
 
 
 move 
 0.45 
 0.03 
 0.10 
 0.06 
 0.15 
 0.49 
 0.19 
 0.22 
 0.00 
 0.02 
 0.01 
 0.20 
 0.40 
 0.00 
 0.04 
 0.06 
 0.20 
 0.04 
 0.03 
 0.00 
 
 
 navigate 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 open 
 0.17 
 0.01 
 0.07 
 0.01 
 0.19 
 0.66 
 0.21 
 0.18 
 0.00 
 0.01 
 0.05 
 0.12 
 0.21 
 0.00 
 0.08 
 0.01 
 0.08 
 0.01 
 0.04 
 0.00 
 
 
 push 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.08 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 scramble 
 0.34 
 0.06 
 0.07 
 0.32 
 0.12 
 0.43 
 0.15 
 0.17 
 0.00 
 0.01 
 0.01 
 0.25 
 0.35 
 0.01 
 0.01 
 0.08 
 0.63 
 0.10 
 0.03 
 0.00 
 
 
 stack 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.07 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
 survive 
 0.01 
 0.00 
 0.01 
 0.02 
 0.07 
 0.49 
 0.08 
 0.01 
 0.00 
 0.00 
 0.06 
 0.05 
 0.07 
 0.00 
 0.02 
 0.01 
 0.01 
 0.00 
 0.03 
 0.00 
 
 
 throw 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.01 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 0.00 
 
 
  Heatmap showing mean human-normalized performance for IMPALA trained for 50M environment steps on a single task and tested on another task. The x-axis and y-axis both list 20 tasks: avoid, bridge, carry, climb, descend, eat, explore, fight, find, gather, hunt, jump, move, navigate, open, push, scramble, stack, survive, throw. The diagonal cells are white, indicating perfect performance on the training task. The highest off-diagonal performance is 0.63 for scramble when trained on stack, and 0.60 for eat when trained on hunt.

### Figure 6

Figure 16: Learning curves for mean human normalized performance on each task, using IMPALA and MT-TB for 500m training runs with the curriculum. The graph shows performance on 17 tasks over 5e8 environment steps. Tasks include eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, and throw. 'eat' and 'avoid' show the highest performance, reaching around 0.7 and 0.6 respectively. Other tasks like 'move' and 'climb' show moderate performance around 0.4. Most other tasks remain below 0.2.

Figure 16: Learning curves for mean human normalized performance on each task, using IMPALA and MT-TB for 500m training runs with the curriculum. The graph shows performance on 17 tasks over 5e8 environment steps. Tasks include eat, avoid, scramble, move, jump, fight, climb, explore, descend, push, stack, open, carry, bridge, hunt, gather, survive, find, navigate, and throw. 'eat' and 'avoid' show the highest performance, reaching around 0.7 and 0.6 respectively. Other tasks like 'move' and 'climb' show moderate performance around 0.4. Most other tasks remain below 0.2.

## References

1. R. Agarwal, M. Schwarzer, P. S. Castro, A. Courville, and M. G. Bellemare. Deep Reinforcement Learning at the Edge of the Statistical Precipice.  arXiv e-prints , page arXiv:2108.13264, Aug. 2021.

2. C. Beattie, J. Z. Leibo, D. Teplyashin, T. Ward, M. Wainwright, H. Küttler, A. Lefrancq, S. Green, V. Valdés, A. Sadik, J. Schrittwieser, K. Anderson, S. York, M. Cant, A. Cain, A. Bolton, S. Gaffney, H. King, D. Hassabis, S. Legg, and S. Petersen. DeepMind Lab.  arXiv e-prints , page arXiv:1612.03801, Dec. 2016.

3. M. G. Bellemare, Y. Naddaf, J. Veness, and M. Bowling. The Arcade Learning Environment: An Evaluation Platform for General Agents.  Journal of Artificial Intelligence Research , 47: 253–279, July 2012.

4. C. Berner, G. Brockman, B. Chan, V. Cheung, P. Dębiak, C. Dennison, D. Farhi, Q. Fischer, S. Hashme, C. Hesse, R. Józefowicz, S. Gray, C. Olsson, J. Pachocki, M. Petrov, H. P. d. O. Pinto, J. Raiman, T. Salimans, J. Schlatter, J. Schneider, S. Sidor, I. Sutskever, J. Tang, F. Wolski,

5. and S. Zhang. Dota 2 with Large Scale Deep Reinforcement Learning.  arXiv e-prints , page arXiv:1912.06680, Dec. 2019.

6. B. Beyret, J. Hernández-Orallo, L. Cheke, M. Halina, M. Shanahan, and M. Crosby. The Animal-AI Environment: Training and Testing Animal-Like Artificial Cognition.  Conference on Neural Information Processing Systems , page 164–176, Sept. 2019.

7. G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba. Openai gym, 2016.

8. N. Brown and T. Sandholm. Superhuman AI for multiplayer poker.  Science , 365(6456): 885–890, Aug. 2019. doi: 10.1126/science.aay2400.

9. K. Cobbe, C. Hesse, J. Hilton, and J. Schulman. Leveraging Procedural Generation to Benchmark Reinforcement Learning.  International Conference on Machine Learning , 119:2048–2056, Dec. 2019.

10. E. Coumans and Y. Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning.  http://pybullet.org , 2016–2021.

11. L. Espeholt, H. Soyer, R. Munos, K. Simonyan, V. Mnih, T. Ward, Y. Doron, V. Firoiu, T. Harley, I. Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In  International Conference on Machine Learning , pages 1407–1416. PMLR, 2018.

12. J. Farebrother, M. C. Machado, and M. Bowling. Generalization and Regularization in DQN.  arXiv e-prints , art. arXiv:1810.00123, Sept. 2018.

13. H. Fu, W. Xu, H. Xue, H. Yang, R. Ye, Y. Huang, Z. Xue, Y. Wang, and C. Lu. RFUniverse: A Physics-based Action-centric Interactive Environment for Everyday Household Tasks.  arXiv e-prints , page arXiv:2202.00199, Jan. 2022.

14. C. Gan, J. Schwartz, S. Alter, D. Mrowca, M. Schrimpf, J. Traer, J. De Freitas, J. Kubilius, A. Bhandwaldar, N. Haber, M. Sano, K. Kim, E. Wang, M. Lingelbach, A. Curtis, K. Feigelis, D. M. Bear, D. Gutfreund, D. Cox, A. Torralba, J. J. DiCarlo, J. B. Tenenbaum, J. H. McDermott, and D. L. K. Yamins. ThreeDWorld: A Platform for Interactive Multi-Modal Physical Simulation.  arXiv e-prints , page arXiv:2007.04954, July 2020.

15. K. Greff, F. Belletti, L. Beyer, C. Doersch, Y. Du, D. Duckworth, D. J. Fleet, D. Gnanapragasam, F. Golemo, C. Herrmann, T. Kipf, A. Kundu, D. Lagun, I. Laradji, Hsueh-Ti, Liu, H. Meyer, Y. Miao, D. Nowrouzezahrai, C. Oztireli, E. Pot, N. Radwan, D. Rebain, S. Sabour, M. S. M. Sajjadi, M. Sela, V. Sitzmann, A. Stone, D. Sun, S. Vora, Z. Wang, T. Wu, K. Moo Yi, F. Zhong, and A. Tagliasacchi. Kubric: A scalable dataset generator.  arXiv e-prints , page arXiv:2203.03570, Mar. 2022.

16. W. H. Guss, B. Houghton, N. Topin, P. Wang, C. Codel, M. Veloso, and R. Salakhutdinov. MineRL: A Large-Scale Dataset of Minecraft Demonstrations.  arXiv e-prints , page arXiv:1907.13440, July 2019.

17. D. Hafner. Benchmarking the Spectrum of Agent Capabilities.  arXiv e-prints , page arXiv:2109.06780, Sept. 2021.

18. D. Hafner, T. P. Lillicrap, J. Ba, and M. Norouzi. Dream to control: Learning behaviors by latent imagination.  CoRR , abs/1912.01603, 2019. URL  http://arxiv.org/abs/1912.01603 .

19. D. Hafner, T. P. Lillicrap, M. Norouzi, and J. Ba. Mastering atari with discrete world models.  CoRR , abs/2010.02193, 2020. URL  https://arxiv.org/abs/2010.02193 .

20. M. Johnson, K. Hofmann, T. Hutton, and D. Bignell. The malmo platform for artificial intelligence experimentation. In  International Joint Conferences on Artificial Intelligence , 2016.

21. D. R. Jones, M. Schonlau, and W. J. Welch. Efficient global optimization of expensive black-box functions.  Journal of Global optimization , 13(4):455–492, 1998.

22. R. V. Juan Linietsky, Ariel Manzur. Godot engine.  https://godotengine.org/ .

23. A. Juliani, A. Khalifa, V.-P. Berges, J. Harper, E. Teng, H. Henry, A. Crespi, J. Togelius, and D. Lange. Obstacle Tower: A Generalization Challenge in Vision, Control, and Planning.  arXiv e-prints , page arXiv:1902.01378, Feb. 2019.

24. N. Justesen, R. Rodriguez Torrado, P. Bontrager, A. Khalifa, J. Togelius, and S. Risi. Illuminating Generalization in Deep Reinforcement Learning through Procedural Level Generation.  arXiv e-prints , page arXiv:1806.10729, June 2018.

25. I. Kanitscheider, J. Huizinga, D. Farhi, W. Hebgeng Guss, B. Houghton, R. Sampedro, P. Zhokhov, B. Baker, A. Ecoffet, J. Tang, O. Klimov, and J. Clune. Multi-task curriculum learning in a complex, visual, hard-exploration domain: Minecraft.  arXiv e-prints , page arXiv:2106.14876, June 2021.

26. H. Küttler, N. Nardelli, T. Lavril, M. Selvatici, V. Sivakumar, T. Rocktäschel, and E. Grefenstette. TorchBeast: A PyTorch Platform for Distributed RL.  arXiv preprint arXiv:1910.03552 , 2019. URL  https://github.com/facebookresearch/torchbeast .

27. H. Küttler, N. Nardelli, A. H. Miller, R. Raileanu, M. Selvatici, E. Grefenstette, and T. Rocktäschel. The NetHack Learning Environment.  arXiv e-prints , page arXiv:2006.13760, June 2020.

28. Q. Li, Z. Peng, L. Feng, Q. Zhang, Z. Xue, and B. Zhou. MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning.  arXiv e-prints , art. arXiv:2109.12674, Sept. 2021.

29. V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra, S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning.  Nature , 518(7540):529–533, Feb. 2015. doi: 10.1038/nature14236.

30. Open Ended Learning Team, A. Stooke, A. Mahajan, C. Barros, C. Deck, J. Bauer, J. Sygnowski, M. Trebacz, M. Jaderberg, M. Mathieu, N. McAleese, N. Bradley-Schmieg, N. Wong, N. Porcel, R. Raileanu, S. Hughes-Fitt, V. Dalibard, and W. M. Czarnecki. Open-Ended Learning Leads to Generally Capable Agents.  arXiv e-prints , page arXiv:2107.12808, July 2021.

31. OpenAI, I. Akkaya, M. Andrychowicz, M. Chociej, M. Litwin, B. McGrew, A. Petron, A. Paino, M. Plappert, G. Powell, R. Ribas, J. Schneider, N. Tezak, J. Tworek, P. Welinder, L. Weng, Q. Yuan, W. Zaremba, and L. Zhang. Solving Rubik’s Cube with a Robot Hand.  arXiv e-prints , art. arXiv:1910.07113, Oct. 2019.

32. E. Parisotto, J. Lei Ba, and R. Salakhutdinov. Actor-Mimic: Deep Multitask and Transfer Reinforcement Learning.  arXiv e-prints , page arXiv:1511.06342, Nov. 2015.

33. A. Petrenko, E. Wijmans, B. Shacklett, and V. Koltun. Megaverse: Simulating Embodied Agents at One Million Experiences per Second.  arXiv e-prints , page arXiv:2107.08170, July 2021.

34. R. Portelas, C. Colas, L. Weng, K. Hofmann, and P. Oudeyer. Automatic curriculum learning for deep RL: A short survey. In C. Bessiere, editor,  Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, IJCAI 2020 , pages 4819–4825. ijcai.org, 2020. doi: 10.24963/ijcai.2020/671. URL  https://doi.org/10.24963/ijcai.2020/671 .

35. A. Puigdomènech Badia, B. Piot, S. Kapturowski, P. Sprechmann, A. Vitvitskyi, D. Guo, and C. Blundell. Agent57: Outperforming the Atari Human Benchmark.  International Conference on Machine Learning , page 507–517, Mar. 2020.

36. A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann. Stable-baselines3: Reliable reinforcement learning implementations.  Journal of Machine Learning Research , 22 (268):1–8, 2021. URL  http://jmlr.org/papers/v22/20-1364.html .

37. A. A. Rusu, S. Gomez Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell. Policy Distillation.  arXiv e-prints , page arXiv:1511.06295, Nov. 2015.

38. M. Samvelyan, R. Kirk, V. Kurin, J. Parker-Holder, M. Jiang, E. Hambro, F. Petroni, H. Küttler, E. Grefenstette, and T. Rocktäschel. MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research.  arXiv e-prints , page arXiv:2109.13202, Sept. 2021.

39. M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra. Habitat: A Platform for Embodied AI Research.  arXiv e-prints , page arXiv:1904.01201, Apr. 2019.

40. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms.  arXiv preprint arXiv:1707.06347 , 2017.

41. D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T. Hubert, L. Baker, M. Lai, A. Bolton, Y. Chen, T. Lillicrap, F. Hui, L. Sifre, G. van den Driessche, T. Graepel, and D. Hassabis. Mastering the game of Go without human knowledge.  Nature , 550(7676):354–359, Oct. 2017. doi: 10.1038/nature24270.

42. S. Srivastava, C. Li, M. Lingelbach, R. Martín-Martín, F. Xia, K. Vainio, Z. Lian, C. Gokmen, S. Buch, C. K. Liu, S. Savarese, H. Gweon, J. Wu, and L. Fei-Fei. BEHAVIOR: Benchmark for Everyday Household Activities in Virtual, Interactive, and Ecological Environments.  arXiv e-prints , page arXiv:2108.03332, Aug. 2021.

43. D. Wierstra, T. Schaul, T. Glasmachers, Y. Sun, J. Peters, and J. Schmidhuber. Natural evolution strategies.  The Journal of Machine Learning Research , 15(1):949–980, 2014.

44. F. Xia, A. R. Zamir, Z.-Y. He, A. Sax, J. Malik, and S. Savarese. Gibson env: real-world perception for embodied agents. In  Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on . IEEE, 2018.

45. T. Yu, D. Quillen, Z. He, R. Julian, A. Narayan, H. Shively, A. Bellathur, K. Hausman, C. Finn, and S. Levine. Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning.  arXiv e-prints , page arXiv:1910.10897, Oct. 2019.

46. A. Zhang, N. Ballas, and J. Pineau. A Dissection of Overfitting and Generalization in Continuous Reinforcement Learning.  arXiv e-prints , art. arXiv:1806.07937, June 2018.

47. C. Zhang, O. Vinyals, R. Munos, and S. Bengio. A Study on Overfitting in Deep Reinforcement Learning.  arXiv e-prints , art. arXiv:1804.06893, Apr. 2018.
