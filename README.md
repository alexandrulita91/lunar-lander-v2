# LunarLander-v2
Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.

## OpenAI Gym
OpenAI Gym is a toolkit for developing and comparing reinforcement learning algorithms. It supports teaching agents everything from walking to playing games like pong or pinball. Gym is an open source interface to reinforcement learning tasks.

## Requirements
- [Python 3.6 or 3.7](https://www.python.org/downloads/release/python-360/)
- [CUDA Toolkit 10.1](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [cuDNN v7.6.5](https://developer.nvidia.com/cuda-10.1-download-archive-base)
- [Pipenv](https://pypi.org/project/pipenv/)
- [Visual C++ 2015 Build Tools](http://go.microsoft.com/fwlink/?LinkId=691126&fixForIE=.exe.)

## How to install the packages
You can install the required Python packages using the following command:
- `pipenv sync`

## Deep Q-learning with Experience Replay
A deep Q network (DQN) (Mnih et al., 2013) is an extension of Q learning, which is a typical deep reinforcement learning method. In DQN, a Q function expresses all action values under all states, and it is approximated using a convolutional neural network. Using the approximated Q function, an optimal policy can be derived. In DQN, a target network, which calculates a target value and is updated by the Q function at regular intervals, is introduced to stabilize the learning process. In DQN, learning is stabilized through a heuristic called experience replay (Lin, 1993) and the use of a target network. Experience replay is a technique that saves time-series data in a buffer called replay memory.

## How to run it
You can run the script using the following command: 
- `pipenv run python lunar_lander_v2_dqn.py`

## Demo video
https://www.youtube.com/watch?v=PEhddjD6QCY

## Improvement ideas
- improve the code quality
- remove unnecessary comments
