# pysc2_GA3C
This is a reinforcement learning agent in pysc2 environment. It's based on [GA3C](https://github.com/NVlabs/GA3C)
.

[![StarCraft 2 MoveToBeacon Evalution](http://img.youtube.com/vi/N8FqFnDF4uM/0.jpg)](http://www.youtube.com/watch?v=N8FqFnDF4uM "StarCraft 2 MoveToBeacon Evalution")

[![StarCraft 2 DefeatRoaches Evalution](http://img.youtube.com/vi/5LJd_5Y6g_o/0.jpg)](http://www.youtube.com/watch?v=5LJd_5Y6g_o "StarCraft 2 DefeatRoaches Evalution")

Note: This agent could reach 25 mean score on MoveToBeacon mini-game (which is good), but on DefeatRoaches it could only get around 60 mean score (No matter using Atari-net or FullyConv-net). This may caused by bad hyper-parameters or off-policy update during training. However, the throughput is better than single-machine A3C and batched A2C.

## Requirements
- python 3 or above
- pysc2 2.0.1
- tensorflow or tensorflow-gpu >= 1.8.0

## Running the Code
1. Issue <code> sh _clean.sh </code> to clean the saved checkpoints of early experiments. (Make sure you change the directory name if you want to keep the checkpoints)
2. Run <code> sh_train.sh </code> command to start training.
3. You can change experiement parameters in <code> Config.py</code>.
>- SC2_MAP_NAME: The map to train on
>- IMAGE_SIZE: The image length of feature maps
>- OPTIMIZER: The optimizer used in training
>- LEARNING_RATE_START, LEARNING_RATE_END: Beginning learning rate and the learning rate in the end


## Training Result

## Reference
[Reinforcement Learning thorugh Asynchronous Advantage Actor-Critic on a GPU](https://openreview.net/forum?id=r1VGvBcxl)

[StarCraft II: A New Challenge for Reinforcement Learning](https://deepmind.com/documents/110/sc2le.pdf)