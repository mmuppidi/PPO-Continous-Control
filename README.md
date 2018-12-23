# Project 2: Continuous Control

The goal of this project is to train an agent to control a double-jointed arm to target locations from Unity ML-Agents toolkit Bananas environment.

![Trained agent](./docs/fast.gif)

### Environment:

The environment consists 20 double-jointed arms which get a goal location at every timestep. Every timestep the arm stays within goal bounds agent receives a +1 reward. The goal of the agent is to follow the goal locations at every timestep and collect as many rewards as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1. The task is complete when agents get an average score of +30 (over 100 consecutive episodes, and over all agents)

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the root directory of this repository, and unzip (or decompress) the file. 

### Training the agent

In order to train the agent, open dist_ppo/Continous_Control.ipynb and run all the cells. 

### Run a trained agent

In order to run a pretrained agent, run all the cells in the Report.ipynb excpet for the training section.

####  For Jupyter notebook newbies

To open the Report.ipynb use the following command from the root of this repository

```
jupyter notebook
```

A webpage will be opened on your browser, click on Report.ipynb, which opens a new page. From here you should be able to run a cell or run everything at once.

#### Requirments

Packages necessary for this project.

``` pip install -r requirements.txt```


#### References

    - [60 Days RL Challenge](https://github.com/andri27-ts/60_Days_RL_Challenge)