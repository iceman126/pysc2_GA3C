# pysc2_GA3C
This is a reinforcement learning agent in pysc2 environment. It's based on [GA3C](https://github.com/NVlabs/GA3C)
.
## Requirements
- python 3 or above
- pysc2 2.0.1
- tensorflow or tensorflow-gpu >= 1.8.0

## Running the Code
1. Issue <code> sh _clean.sh </code> to clean the saved checkpoints of early experiments. (Make sure you change the directory name if you want to keep the checkpoints)
2. Run <code> sh_train.sh </code> command to start training.
3. You can change experiement parameters in <code> Config.py</code>. (I will describe more details ASAP)

## Training Result

## Reference
[Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://openreview.net/forum?id=r1VGvBcxl)
[StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf)