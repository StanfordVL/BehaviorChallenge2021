BEHAVIOR Challenge @ ICCV2021
=============================================

This repository contains starter code for BEHAVIOR Challenge 2021 brought to you by [Stanford Vision and Learning Lab](http://svl.stanford.edu).
For an overview of the challenge and the workshop and information about the tasks, evaluation metrics, datasets and setup, visit [the challenge website](https://behavior.stanford.edu/).

For more information or questions, contact us at behavior.benchmark@gmail.com

Participation Guidelines
-----------------------------

In the following, we summarize the most relevant points to participate. For a full description, visit the online [BEHAVIOR Participation Guidelines](https://docs.google.com/document/d/1u2m9Ld6Qo3eG-fvCzuAZN6lHxwpVVBlwLVdzo_WDlNI/edit#heading=h.7h8v0rnlggnt). 

Participate in the contest by registering on the [EvalAI challenge page](https://eval.ai/web/challenges/challenge-page/1190/overview) and creating a team. In the Minival and Evaluation phases, participants will upload docker containers with their agents that evaluated on a AWS GPU-enabled instance. Before pushing the submissions for remote evaluation, participants should test the submission docker locally to make sure it is working. Instructions for training, local evaluation, and online submission are provided below.

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
  ```Dockerfile
  FROM igibson/behavior_challenge_2021:latest
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

Our BEHAVIOR Challenge 2021 consists of four phases:

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
