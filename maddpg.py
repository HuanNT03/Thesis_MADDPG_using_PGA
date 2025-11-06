# maddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# SIMPLIFIED POWERFUL Actor Network 
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        
        # Simpler but effective architecture
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)
        
        # Simple but effective normalization
        self.ln1 = nn.LayerNorm(256)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        
        # ReLU works well for this simpler network
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Less dropout
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, state):
        x = self.activation(self.ln1(self.fc1(state)))
        x = self.dropout(x)
        
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.activation(self.ln3(self.fc3(x)))
        
        # Output with tanh activation for bounded actions
        action = torch.tanh(self.fc4(x))
        return action

# SIMPLIFIED POWERFUL Critic Network
class Critic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim):
        super(Critic, self).__init__()
        
        # Critic processes all agents' states and actions
        input_dim = (state_dim + action_dim) * num_agents
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Simpler but effective architecture
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 1)
        
        # Simple normalization
        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(64)
        
        # ReLU activation
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # Less dropout
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, states, actions):
        # Concatenate all states and actions
        x = torch.cat(states + actions, dim=1)
        
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.dropout(x)
        
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        
        x = self.activation(self.ln3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.activation(self.ln4(self.fc4(x)))
        
        q_value = self.fc5(x)
        return q_value

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, agent_id, num_agents, lr_actor=0.0015, lr_critic=0.003, gamma=0.99, tau=0.01, device=None, env_params=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.device = device if device is not None else torch.device("cpu")
        
        # Gradient clipping
        self.max_grad_norm = 5.0
        
        # Improved exploration strategy
        self.epsilon_start = 0.95  # Very high initial exploration  
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.9985  # Slower decay
        self.epsilon = self.epsilon_start
        
        # OU noise - start higher, decay over time
        self.noise_scale_start = 0.3
        self.noise_scale_end = 0.05
        self.noise_scale = self.noise_scale_start
        self.noise_theta = 0.15
        self.noise_sigma = 0.25
        self.noise_state = np.zeros(action_dim)
        
        # Expert guidance (Greedy strategy for imitation)
        self.env_params = env_params
        self.use_expert = True
        self.expert_epsilon = 0.5  # Probability to use expert in early training
        self.expert_decay = 0.997
        
        # Training tracking
        self.training_step = 0

        # Tạo mạng Actor và Critic, cùng với các mạng target của chúng
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(num_agents, state_dim, action_dim).to(self.device)
        self.target_critic = Critic(num_agents, state_dim, action_dim).to(self.device)

        # Sao chép trọng số ban đầu
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Simple but effective optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def get_expert_action(self, state):
        """Get greedy expert action for imitation learning"""
        if self.env_params is None:
            return None
            
        # Decode state (same as in GreedyStrategy)
        g_su_du = 10**state[1]
        g_su_rbs = 10**state[2]
        g_rbs_du = 10**state[3]
        g_jam_su = 10**state[4]
        g_jam_du = 10**state[5]
        g_jam_rbs = 10**state[6]
        
        pga_choice = self.env_params['pga_gain']
        
        # Calculate SINR for D2D
        num_d2d = g_jam_su * g_su_du * (pga_choice**2) * self.env_params['jammer_power']
        den_d2d = g_jam_du * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr_d2d = num_d2d / (den_d2d + 1e-9)
        
        # Calculate SINR for Relay
        num1_relay = g_jam_su * g_su_rbs * (pga_choice**2) * self.env_params['jammer_power']
        den1_relay = g_jam_rbs * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr1_relay = num1_relay / (den1_relay + 1e-9)
        
        num2_relay = g_jam_rbs * g_rbs_du * (pga_choice**2) * self.env_params['jammer_power']
        den2_relay = g_jam_du * self.env_params['jammer_power'] + self.env_params['noise_power']
        sinr2_relay = num2_relay / (den2_relay + 1e-9)
        
        sinr_relay = min(sinr1_relay, sinr2_relay)
        
        # Choose best mode
        if sinr_d2d >= sinr_relay:
            # D2D mode: action[0]=1 (send bit 1), action[1]=-1 (D2D)
            expert_action = np.array([1.0, -1.0])
        else:
            # Relay mode: action[0]=1, action[1]=1 (Relay)
            expert_action = np.array([1.0, 1.0])
            
        return expert_action
    
    def select_action(self, state, add_noise=True):
        state_np = state
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Early training: use expert with probability
        if add_noise and self.use_expert and self.env_params is not None and np.random.rand() < self.expert_epsilon:
            action = self.get_expert_action(state_np)
            if action is not None:
                # Add small noise to expert action for exploration
                action += np.random.normal(0, 0.1, self.action_dim)
                self.training_step += 1
                return np.clip(action, -1, 1)
        
        # Normal epsilon-greedy exploration
        if add_noise and np.random.rand() < self.epsilon:
            action = np.random.uniform(-1, 1, self.action_dim)
            # Bias towards D2D and sending bit 1
            action[0] = np.random.uniform(0, 1)  # Bias to send bit 1
            action[1] = np.random.uniform(-1, 0)  # Bias to D2D
        else:
            # Use actor network
            self.actor.eval()
            with torch.no_grad():
                action = self.actor(state_tensor).squeeze(0).cpu().numpy()
            self.actor.train()
            
            if add_noise:
                # OU noise
                dx = self.noise_theta * (0 - self.noise_state) + self.noise_sigma * np.random.randn(self.action_dim)
                self.noise_state += dx
                noise = self.noise_scale * self.noise_state
                action += noise
        
        self.training_step += 1
        return np.clip(action, -1, 1)
    
    def reset_noise(self):
        """Reset noise state and decay epsilon for new episode"""
        self.noise_state = np.zeros(self.action_dim)
        # Decay epsilon each episode
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # Decay expert epsilon
        self.expert_epsilon = max(0.0, self.expert_epsilon * self.expert_decay)
        # Decay noise scale
        self.noise_scale = max(self.noise_scale_end, self.noise_scale * 0.995)

    def update_targets(self):
        # Soft update cho các mạng target
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))