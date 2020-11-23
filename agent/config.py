import torch
### 可以在这里写入 最多有多少飞机满足条件 即red_dqn_agent.py中的maxppp
in_c = 20
out_c = 20
decay = 1
LR = 0.001
a2a_LX11 = 20
layers = 4
gamma = 0.5
n_states = 20
extension = 2
batch_size = 1
actions = out_c
epsilon = 0.299
checkpoints = None
buffer_size = 10000
device = "cuda" if torch.cuda.is_available() else "cpu"
