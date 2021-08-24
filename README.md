BEHAVIOR Challenge @ ICCV2021 Embodied AI Workshop
=============================================

This repository contains starter code for BEHAVIOR Challenge 2021 brought to you by [Stanford Vision and Learning Lab](http://svl.stanford.edu).
For an overview of the challenge and the workshop, visit [the challenge website](http://svl.stanford.edu/behavior/challenge.html).

Tasks
----------------------------
Please visit [the challenge website](http://svl.stanford.edu/behavior/challenge.html).
<!-- The iGibson Challenge 2021 uses the iGibson simulator [1] and is composed of two navigation tasks that represent important skills for autonomous visual navigation:

Interactive Navigation            |  Social Navigation
:-------------------------:|:-------------------------:
<img src="images/cvpr21_interactive_nav.png" height="400"> | <img src="images/cvpr21_social_nav.png" height="400">

- **Interactive Navigation**: the agent is required to reach a navigation goal specified by a coordinate (as in PointNav [2]) given visual information (RGB+D images). The agent is allowed (or even encouraged) to collide and interact with the environment in order to push obstacles away to clear the path. Note that all objects in our scenes are assigned realistic physical weight and fully interactable. However, as in the real world, while some objects are light and movable by the robot, others are not. Along with the furniture objects originally in the scenes, we also add additional objects (e.g. shoes and toys) from the Google Scanned Objects dataset to simulate real-world clutter. We will use Interactive Navigation Score (INS) [3] to evaluate agents' performance in this task.

- **Social Navigation**: the agent is required to navigate the goal specified by a coordinate while moving around pedestrians in the environment. Pedestrians in the scene move towards randomly sampled locations, and their movement is simulated using the social-forces model ORCA [4] integrated in iGibson [1], similar to the simulation enviroments in [5]. The agent shall avoid collisions or proximity to pedestrians beyond a threshold (distance <0.3 meter) to avoid episode termination. It should also maintain a comfortable distance to pedestrians (distance <0.5 meter), beyond which the score is penalized but episodes are not terminated. We will use the average of STL (Success weighted by Time Length) and PSC (Personal Space Compliance) to evaluate the agents' performance. More details can be found in the "Evaluation Metrics" section below. -->


Evaluation Metrics
-----------------------------
Please visit [the challenge website](http://svl.stanford.edu/behavior/challenge.html).
<!-- - **Interactive Navigation**: We will use Interactive Navigation Score (INS) as our evaluation metrics. INS is an average of Path Efficiency and Effort Efficiency. Path Efficiency is equivalent to SPL (Success weighted by Shortest Path). Effort Efficiency captures both the excess of displaced mass (kinematic effort) and applied force (dynamic effort) for interaction with objects. We argue that the agent needs to strike a healthy balance between taking a shorter path to the goal and causing less disturbance to the environment. More details can be found in [our paper](https://ieeexplore.ieee.org/abstract/document/8954627/).

- **Social Navigation**: We will use the average of STL (Success weighted by Time Length) and PSC (Personal Space Compliance) as our evaluation metrics. STL is computed by success * (time_spent_by_ORCA_agent / time_spent_by_robot_agent). The second term is the number of timesteps that an oracle ORCA agent take to reach the same goal assigned to the robot. This value is clipped by 1. In the context of Social Navigation, we argue STL is more applicable than SPL because a robot agent can achieve perfect SPL by "waiting out" all pedestrians before it makes a move, which defeats the purpose of the task. PSC (Personal Space Compliance) is computed as the percentage of timesteps that the robot agent comply with the pedestrians' personal space (distance >= 0.5 meter). We argue that the agent needs to strike a heathy balance between taking a shorted time to reach the goal and incuring less personal space violation to the pedestrians. -->

Dataset
----------------------------
Please visit [the challenge website](http://svl.stanford.edu/behavior/challenge.html).
<!-- We provide 8 scenes reconstructed from real world apartments in total for training in iGibson. All objects in the scenes are assigned realistic weight and fully interactable. For interactive navigation, we also provide 20 additional small objects (e.g. shoes and toys) from the Google Scanned Objects dataset. For fairness, please only use these scenes and objects for training.

For evaluation, we have 2 unseen scenes in our **dev** split and 5 unseen scenes in our **test** split. We also use 10 unseen small objects (they will share the same object categories as the 20 training small objects, but they will be different object instances).

Visualizations for the 8 training scenes.

![alt text](images/cvpr21_dataset.gif) -->


Setup
----------------------------
Please visit [the challenge website](http://svl.stanford.edu/behavior/challenge.html).
<!-- We adopt the following task setup:

- **Observation**: (1) Goal position relative to the robot in polar coordinates, (2) current linear and angular velocities, (3) RGB+D images.
- **Action**: Desired normalized linear and angular velocity.
- **Reward**: We provide some basic reward functions for reaching goal and making progress. Feel free to create your own.
- **Termination conditions**: The episode termintes after 500 timesteps or the robot collides with any pedestrian in the Social Nav task.

The tech spec for the robot and the camera sensor can be found in [here](Parameters.md).

For **Interactive Navigation**, we place N additional small objects (e.g. toys, shoes) near the robot's shortest path to the goal (N is proportional to the path length). These objects are generally physically lighter than the objects originally in the scenes (e.g. tables, chairs).

For **Social Navigation**, we place M pedestrians randomly in the scenes that pursue their own random goals during the episode while respecting each other's personal space (M is proportional to the physical size of the scene). The pedestrians have the same maximum speed as the robot. They are aware of the robot so they won't walk straight into the robot. However, they also won't yield to the robot: if the robot moves straight towards the pedestrians, it will hit them and the episode will fail. -->

Participation Guidelines
-----------------------------
Participate in the contest by registering on the EvalAI challenge page and creating a team. Participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

### Local Evaluation
- Step 1: Clone the challenge repository
  ```bash
  git clone https://github.com/StanfordVL/BehaviorChallenge2021.git
  cd BehaviorChallenge2021
  ```

  Two example agents are provided in `simple_agent.py` and `rl_agent.py`: `RandomAgent` and `PPOAgent`.
  We also provide randomly initialized checkpoints for `PPOAgent`, stored in `checkpoints/`.
  Please implement your own agent and instantiate it from `agent.py`.

- Step 2: Install nvidia-docker2, following the guide: https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0). 

- Step 3: Modify the provided Dockerfile to accommodate any dependencies. A minimal Dockerfile is shown below.
  TODO: add the correct base docker 
  ```Dockerfile
  FROM gibsonchallenge/gibson_challenge_2021:latest
  ENV PATH /miniconda/envs/gibson/bin:$PATH

  ADD agent.py /agent.py
  ADD simple_agent.py /simple_agent.py
  ADD rl_agent.py /rl_agent.py

  ADD submission.sh /submission.sh
  WORKDIR /
  ```

  Then build your docker container with `docker build . -t my_submission` , where `my_submission` is the docker image name you want to use.

- Step 4: 

  Download the challenge data by completing the user agreement (https://forms.gle/ecyoPtEcCBMrQ3qF9). Place `ig_dataset` and `igibson.key` under `BehaviorChallenge2021`.

- Step 5:

  Evaluate locally:

  You can run `./test_minival_locally.sh --docker-name my_submission`
  
  <!-- The script by default evaluates Social Navigation. If you want to evaluate Interactive Navigation, you need to change `CONFIG_FILE`, `TASK` and `EPISODE_DIR` in the script and make them consistent. It's recommended that you use TASK environment variable to switch agents in `agent.py` if you intend to use different policies for these two tasks. -->

### Online submission
Follow instructions in the submit tab of the EvalAI challenge page to submit your docker image. Note that you will need a version of EvalAI >= 1.2.3. Here we reproduce part of those instructions for convenience:

```bash
# Installing EvalAI Command Line Interface
pip install "evalai>=1.2.3"

# Set EvalAI account token
evalai set_token <your EvalAI participant token>

# Push docker image to EvalAI docker registry
evalai push my_submission:latest --phase <phase-name>
```

The valid challenge phases are: `behavior-minival-onboard-sensing-1190`, `behavior-minival-full-observability-1190`, `behavior-dev-onboard-sensing-1190`, `behavior-dev-full-observability-1190`, `behavior-test-onboard-sensing-1190`, `behavior-test-full-observability-1190`.

Our iGibson Challenge 2021 consists of four phases:

- Minival Phase (`behavior-minival-onboard-sensing-1190`, `behavior-minival-full-observability-1190`): The purpose of this phase to make sure your policy can be successfully submitted and evaluated. Participants are expected to download our starter code and submit a baseline policy, even a trivial one, to our evaluation server to verify their entire pipeline is correct. The submission will only be evaluated on one activity.
- Dev Phase (`behavior-dev-onboard-sensing-1190`, `behavior-dev-full-observability-1190`): This phase is split into Onboard Sensing and Full Observability tracks. Participants are expected to submit their solutions to **each** of the tasks separately because they have different observation spaces. The results will be evaluated on the dataset **dev** split and the leaderboard will be updated within 24 hours.
- Test Phase (`behavior-test-onboard-sensing-1190`, `behavior-test-full-observability-1190`): This phase is also split into Onboard Sensing and Full Observability tracks. Participants are expected to submit a maximum of 5 solutions during the last few weeks of the challenge. The solutions will be evaluated on the dataset **test split** and the results will NOT be made available until the end of the challenge.
<!-- - Winner Demo Phase: To increase visibility, the best three entries of each task of our challenge will have the opportunity to showcase their solutions in live or recorded video format during CVPR2021! All the top runners will be able to highlight their solutions and findings to the CVPR audience. Feel free to check out [our presentation](https://www.youtube.com/watch?v=0BvUSjcc0jw&list=PL4XI7L9Xv5fVUMEb1eYOaH8y1b6j8xiMM) and [our participants' presentations](https://www.youtube.com/watch?v=NBE-iXpyCCU&list=PL4XI7L9Xv5fVULPNAqiGQ2yK07k78-02h) from our challenge last year on YouTube. -->


### Training
#### Using Docker
Train with minival split (with only one of the activities): `./train_minival_locally.sh --docker-name my_submission`.

Note that due to the difficulty of BEHAVIOR activities, the default training with PPO will NOT converge to success. We provide this training pipeline just as a starting point for participants to further build upon.

#### Not using Docker
- Step 0: install [anaconda](https://docs.anaconda.com/anaconda/install/) and create a python3.6 environment
  ```
  conda create -n igibson python=3.6
  conda activate igibson
  ```
- Step 1: install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). We tested with CUDA 10.0 and 10.1 and cuDNN 7.6.5

- Step 2: install EGL dependency
  ```
  sudo apt-get install libegl1-mesa-dev
  ```
- Step 3: install [iGibson](http://svl.stanford.edu/igibson/) **from source** by following the [documentation](http://svl.stanford.edu/igibson/docs).

- Step 4: Download the challenge data by completing the user agreement (https://forms.gle/ecyoPtEcCBMrQ3qF9), and place `ig_dataset` and `igibson.key` under `igibson/data`.

- Step 5: start training with stable-baselines3!
  ```
  cd iGibson/igibson/examples/demo
  python stable_baselines3_behavior_example.py
  ```
  This will train with one activity in one scene, defined in `behavior_onboard_sensing.yaml`. 
  
Feel free to skip Step 5 if you want to use other frameworks for training. This is just a example starter code for your reference.


<!-- References 
-------------------
[1] [iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes](https://arxiv.org/abs/2012.02924). Bokui Shen, Fei Xia, Chengshu Li, Roberto Martín-Martín, Linxi Fan, Guanzhi Wang, Shyamal Buch, Claudia D'Arpino, Sanjana Srivastava, Lyne P Tchapmi, Micael E Tchapmi, Kent Vainio, Li Fei-Fei, Silvio Savarese. Preprint arXiv:2012.02924, 2020.

[2] [On evaluation of embodied navigation agents](https://arxiv.org/abs/1807.06757). Peter Anderson, Angel Chang, Devendra Singh Chaplot, Alexey Dosovitskiy, Saurabh Gupta, Vladlen Koltun, Jana Kosecka, Jitendra Malik, Roozbeh Mottaghi, Manolis Savva, Amir R. Zamir. arXiv:1807.06757, 2018.

[3] [Interactive Gibson: A Benchmark for Interactive Navigation in Cluttered Environments](https://ieeexplore.ieee.org/abstract/document/8954627/).  Xia, Fei, William B. Shen, Chengshu Li, Priya Kasimbeg, Micael Tchapmi, Alexander Toshev, Roberto Martín-Martín, and Silvio Savarese. arXiv preprint arXiv:1910.14442 (2019).

[4] [RVO2 Library: Reciprocal Collision Avoidance for Real-Time Multi-Agent Simulation](https://gamma.cs.unc.edu/RVO2/). Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, and Dinesh Manocha, 2011.

[5] [Robot Navigation in Constrained Pedestrian Environments using Reinforcement Learning](https://arxiv.org/abs/2010.08600) Claudia Pérez-D'Arpino, Can Liu, Patrick Goebel, Roberto Martín-Martín and Silvio Savarese. Preprint arXiv:2010.08600, 2020. -->
