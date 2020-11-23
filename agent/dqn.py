import torch
import agent.config as  config
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self,in_channel, out_channel, layers = 4, extension = 2):
        super(Network, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.head1 = nn.Sequential(nn.Linear(in_channel, 32),nn.ReLU(inplace=True))
        self.head2 = nn.Sequential(nn.Linear(in_channel, 32), nn.ReLU(inplace=True))
        self.head3 = nn.Sequential(nn.Linear(in_channel, 32), nn.ReLU(inplace=True))
        i, op = self._body_layers(layers, extension)
        self.body = nn.Sequential(*op)
        self.tail = nn.Linear(i, out_channel)

    def forward(self, obs,obs2, obs3, mask):
        input1 = self.head1(obs)
        input2 = self.head2(obs2)
        input3 = self.head3(obs3)
        f1 = torch.cat((input1, input2, input3),1)
        f2 = self.body(f1)
        actions = self.tail(f2)

        actions = actions - actions.min() + 1
        #这里是为了全变成正的
        actions = actions * mask
        print(actions.data)
        return actions

    def _body_layers(self,layers = 4, extension = 2):
        i, o = 96, 32 * extension
        op = []
        mid_layers = layers - 2
        for layer in range(mid_layers):
            op.append(nn.Linear(i, o))
            op.append(nn.ReLU(inplace=True))
            times = 2 if layer < mid_layers // 2 - 1 else 0.5
            i, o = o, int(o * times)
        return i, op

net = Network(in_channel=config.in_c, out_channel=config.out_c, layers = 10)


class DQN(nn.Module):
    def __init__(self, gamma = config.gamma, epsilon = config.epsilon,
                 decay = config.decay, checkpoints = config.checkpoints):
        super(DQN, self).__init__()
        self.update_model, self.target_model = \
            self._create_model(checkpoints),self._create_model(checkpoints)
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.log = {"Episode":[], "Episode_Reward":[], "Loss":[]}
        self.step_count = 0
        self.buffer_count = 0
        self.copy_internal = 100
        self.save_internal = 1000
        self.buffer = np.zeros((config.buffer_size, 4*config.n_states * 2 + 2))
        #buffer 的结构是[当前状态，action，reward，未来状态]  
        self.optimizer = optim.Adam(self.update_model.parameters(), lr = config.LR)
        self.loss = nn.MSELoss()

    def _choose_action(self,obs, obs2, obs3, mask):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, np.sum(mask)-1)

        obs,obs2, obs3 = self._totensor(obs),self._totensor(obs2),self._totensor(obs3)
        mask = self._totensor(mask)
        q_value = self.update_model.forward(obs, obs2, obs3, mask)
        action = (torch.max(q_value, 1)[1].data.numpy())[0]
        return action

    def _copy_weights(self):
        # 这个函数是把update model的参数复制给target model
        self.target_model.load_state_dict(self.update_model.state_dict())

    def _create_model(self,checkpoints = None):
        # 我们先假定模型文件都是单卡的
        # 且默认文件只包括参数，不包括模型，即用torch.save(net.state_dict(),PATH)的形式保存
        model = Network(config.in_c, config.out_c,config.layers, config.extension)
        #model = model.load_state_dict(torch.load(checkpoints)) if checkpoints is not None else model
        return model

    def _learning(self):
        print("!!!!!!!!!!!!!!!  I am trying to learn something!!!!")
        if self.step_count % self.copy_internal == 0:
            self._copy_weights()
            print("Target model updates!")
        if self.step_count % self.save_internal == 0:
            name = './model/dqn'+str(self.step_count)+'.pth'
            torch.save(self.update_model.state_dict(), name)
        self.step_count += 1
        self.optimizer.zero_grad()
        data_index = np.random.choice(config.buffer_size, config.batch_size)
        transitions = self.buffer[data_index,:]
        batch_states = torch.FloatTensor(transitions[:, :config.n_states*4]).to(config.device)
        batch_actions = torch.LongTensor(transitions[:, config.n_states*4:config.n_states*4+1].astype(int)).to(config.device)
        batch_rewards = torch.FloatTensor(transitions[:, config.n_states*4+1:config.n_states*4+2]).to(config.device)
        batch_states_next = torch.FloatTensor(transitions[:, -config.n_states*4:]).to(config.device)
        q_target = self.target_model.forward(batch_states_next[:, :config.n_states],
                                             batch_states_next[:, config.n_states:config.n_states*2],
                                             batch_states_next[:, config.n_states*2:config.n_states*3],
                                             batch_states_next[:, -config.n_states:]).detach()

        q_learn = self.update_model(batch_states[:, :config.n_states],
                                             batch_states[:, config.n_states:config.n_states*2],
                                             batch_states[:, config.n_states*2:config.n_states*3],
                                             batch_states[:, -config.n_states:]).gather(1, batch_actions)
        q_target = batch_rewards + self.gamma * q_target.max(1)[0].view(config.batch_size,1)
        loss = self.loss(q_learn, q_target)
        loss.backward()
        self.optimizer.step()

    def _store_transition(self, s, a, r, s_):
        # replace the old memory with new memory
        index = self.buffer_count % config.buffer_size
        ok = np.hstack((s, [a, r], s_))

        self.buffer[index, :] = ok
        self.buffer_count += 1

    def _totensor(self, obs):
        obs = [obs] if len(obs.shape) == 1 else obs
        obs = torch.FloatTensor(obs)
        return obs
