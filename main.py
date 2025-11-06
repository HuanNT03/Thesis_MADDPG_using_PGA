# main.py

import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import csv
import os
from datetime import datetime
import time

from environment import CommEnvironment
from maddpg import MADDPGAgent
from baselines import DirectTransmission, GreedyStrategy, FrequencyHopping

# --- OPTIMIZED Hyperparameters with Imitation Learning ---
NUM_EPISODES = 300  # More episodes for imitation + RL
STEPS_PER_EPISODE = 1000  # Good balance for learning
BATCH_SIZE = 256  # Good batch size for simplified networks  
REPLAY_BUFFER_SIZE = 150000  # Larger buffer for diverse experiences
NUM_AGENTS = 5
WARMUP_STEPS = 5000  # Shorter warmup with expert guidance
TRAIN_FREQUENCY = 1  # Train every step when learning from expert
TRAIN_ITERATIONS = 1  # Single iteration per training for stability

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay for better sample efficiency"""
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = 0.001  # Anneal beta to 1
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # New experiences get max priority
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        buffer_len = len(self.buffer)
        if buffer_len == 0:
            return None
        
        # Calculate sampling probabilities
        priorities = self.priorities[:buffer_len]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(buffer_len, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.buffer)

# Keep old ReplayBuffer for compatibility
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class RunningMeanStd:
    """Running mean and standard deviation for reward normalization"""
    def __init__(self, epsilon=1e-4):
        self.mean = 0
        self.var = 1
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

def evaluate_baseline(baseline_agent, env, num_episodes):
    """Hàm để đánh giá hiệu năng của một baseline với đầy đủ metrics."""
    total_rewards = []
    all_metrics = []
    print(f"\n--- Evaluating {baseline_agent.__class__.__name__} ---")

    for episode in range(num_episodes):
        states = env.reset()
        episode_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            # Với FH, ta tính reward trực tiếp thay vì qua env.step
            if isinstance(baseline_agent, FrequencyHopping):
                rewards = baseline_agent.get_rewards(states)
                # Cần cập nhật môi trường để lấy state tiếp theo (dùng action trung tính)
                neutral_actions = [[0.0, 0.0] for _ in range(env.num_agents)]
                _, _, _ = env.step(neutral_actions)
                states = env._get_states()
            else: # Với DT và Greedy
                actions = baseline_agent.select_actions(states)
                next_states, rewards, _ = env.step(actions)
                states = next_states
            
            episode_reward += np.mean(rewards)
        
        total_rewards.append(episode_reward / STEPS_PER_EPISODE)
        metrics = env.get_metrics()
        all_metrics.append(metrics)
        
        if (episode + 1) % (num_episodes // 5) == 0:
            print(f"  Episode {episode + 1}/{num_episodes} finished.")

    avg_metrics = {
        'reward': np.mean(total_rewards),
        'success_rate': np.mean([m['success_rate'] for m in all_metrics]),
        'energy_efficiency': np.mean([m['energy_efficiency'] for m in all_metrics]),
        'avg_energy': np.mean([m['avg_energy_per_step'] for m in all_metrics]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in all_metrics]),
        'throughput_efficiency': np.mean([m['throughput_efficiency'] for m in all_metrics])
    }
    
    return avg_metrics

def train(agents, replay_buffer):
    if len(replay_buffer) < BATCH_SIZE:
        return None

    samples = replay_buffer.sample(BATCH_SIZE)
    device = agents[0].device if agents else torch.device("cpu")
    
    # Tách dữ liệu từ batch
    states, actions, rewards, next_states, dones = zip(*samples)

    # No reward clipping - let the algorithm learn the true reward scale
    # rewards are already well-designed from environment

    # Chuyển đổi sang tensor (tối ưu hóa: gom list các ndarray thành một np.array trước)
    device = agents[0].device if agents else torch.device("cpu")
    states_t = [torch.from_numpy(np.array([s[i] for s in states], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    actions_t = [torch.from_numpy(np.array([a[i] for a in actions], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    rewards_t = [torch.from_numpy(np.array([r[i] for r in rewards], dtype=np.float32)).unsqueeze(1).to(device) for i in range(NUM_AGENTS)]
    next_states_t = [torch.from_numpy(np.array([ns[i] for ns in next_states], dtype=np.float32)).to(device) for i in range(NUM_AGENTS)]
    dones_t = [torch.from_numpy(np.array([d[i] for d in dones], dtype=np.float32)).unsqueeze(1).to(device) for i in range(NUM_AGENTS)]

    avg_actor_loss = 0.0
    avg_critic_loss = 0.0
    avg_actor_grad_norm = 0.0
    avg_critic_grad_norm = 0.0
    avg_q_value = 0.0

    for i, agent in enumerate(agents):
        # --- Cập nhật Critic ---
        with torch.no_grad():
            next_actions_t = [agents[j].target_actor(next_states_t[j]) for j in range(NUM_AGENTS)]
            next_q = agent.target_critic(next_states_t, next_actions_t)
            target_q = rewards_t[i] + agent.gamma * next_q * (1 - dones_t[i])
        
        current_q = agent.critic(states_t, actions_t)
        avg_q_value += current_q.mean().item()
        
        # Simple MSE loss
        critic_loss = nn.MSELoss()(current_q, target_q)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.max_grad_norm)
        avg_critic_grad_norm += critic_grad_norm.item()
        agent.critic_optimizer.step()

        # --- Cập nhật Actor ---
        current_actions_t = actions_t.copy()
        predicted_actions = agent.actor(states_t[i])
        current_actions_t[i] = predicted_actions
        
        # Policy gradient loss
        policy_loss = -agent.critic(states_t, current_actions_t).mean()
        
        # Imitation loss - learn from expert actions
        imitation_loss = 0
        if agent.use_expert and agent.expert_epsilon > 0.1:
            # Get expert actions for this batch
            expert_actions_batch = []
            states_np = states_t[i].cpu().numpy()
            for s in states_np:
                expert_a = agent.get_expert_action(s)
                if expert_a is not None:
                    expert_actions_batch.append(expert_a)
                else:
                    expert_actions_batch.append(np.zeros(agent.action_dim))
            
            expert_actions_tensor = torch.FloatTensor(np.array(expert_actions_batch)).to(device)
            imitation_loss = 0.3 * nn.MSELoss()(predicted_actions, expert_actions_tensor)
        
        # Combined loss
        actor_loss = policy_loss + imitation_loss
        
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), agent.max_grad_norm)
        avg_actor_grad_norm += actor_grad_norm.item()
        agent.actor_optimizer.step()

        # Tích lũy loss để logging
        avg_actor_loss += actor_loss.item()
        avg_critic_loss += critic_loss.item()

    avg_actor_loss /= NUM_AGENTS
    avg_critic_loss /= NUM_AGENTS
    avg_actor_grad_norm /= NUM_AGENTS
    avg_critic_grad_norm /= NUM_AGENTS
    avg_q_value /= NUM_AGENTS

    return avg_actor_loss, avg_critic_loss, avg_actor_grad_norm, avg_critic_grad_norm, avg_q_value

if __name__ == "__main__":
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        # Enable optimized GPU operations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    env = CommEnvironment(num_agents=NUM_AGENTS)
    
    # Prepare environment parameters for expert guidance
    env_params = {
        'pga_gain': env.pga_gain,
        'jammer_power': env.jammer_power,
        'noise_power': env.noise_power,
    }
    
    # Create agents with expert guidance
    agents = [MADDPGAgent(env.state_dim, env.action_dim, i, NUM_AGENTS, 
                         lr_actor=0.0015, lr_critic=0.003, tau=0.01, device=device, env_params=env_params) for i in range(NUM_AGENTS)]
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    episode_rewards = []
    episode_metrics = []
    total_steps = 0
    best_reward = -np.inf
    
    # Additional tracking for detailed monitoring
    actor_grad_norms = []
    critic_grad_norms = []
    q_values = []
    
    print("="*80)
    print("🚀 MADDPG WITH EXPERT GUIDANCE - IMITATION + REINFORCEMENT")
    print("="*80)
    print(f"Training: {NUM_EPISODES} episodes × {STEPS_PER_EPISODE} steps")
    print(f"Batch: {BATCH_SIZE} | Buffer: {REPLAY_BUFFER_SIZE} | Warmup: {WARMUP_STEPS}")
    print(f"LR - Actor: 0.0015, Critic: 0.003 (HIGHER)")
    print(f"Tau: 0.01 | Gamma: 0.99 | Grad Clip: 5.0")
    print(f"SINR Threshold: 0.15 (STANDARD)")
    print(f"Network: 256-256-128 (actor), 512-256-128-64 (critic)")
    print(f"")
    print(f"🔥 BREAKTHROUGH FEATURES:")
    print(f"  • IMITATION LEARNING from Greedy Expert (Bootstrap success!)")
    print(f"  • Expert guidance: 50% → 0% over episodes")
    print(f"  • Behavior cloning loss (0.3x weight)")
    print(f"  • Biased exploration towards expert actions")
    print(f"  • Balanced reward (no env advantage for baselines)")
    print(f"  • Slower epsilon decay for better exploration")
    print(f"")
    print(f"🎯 STRATEGY: Learn from expert, then surpass with RL!")
    print("="*80)
    
    training_start_time = time.time()
    
    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        states = env.reset()
        current_episode_reward = 0
        last_actor_loss = None
        last_critic_loss = None
        last_q_value = None
        
        # Reset noise for new episode
        for agent in agents:
            agent.reset_noise()

        for step in range(STEPS_PER_EPISODE):
            # Select actions with exploration noise
            actions = [agent.select_action(states[i], add_noise=True) for i, agent in enumerate(agents)]
            
            next_states, rewards, dones = env.step(actions)
            
            # Rewards are already well-scaled from environment, just store them
            replay_buffer.push(states, actions, rewards, next_states, dones)
            
            states = next_states
            current_episode_reward += np.mean(rewards)
            total_steps += 1

            # Adaptive training - train more when using expert
            train_freq = TRAIN_FREQUENCY
            if agents[0].expert_epsilon > 0.1:  # When still learning from expert
                train_freq = 1  # Train every step
            
            if total_steps > WARMUP_STEPS and len(replay_buffer) > BATCH_SIZE and total_steps % train_freq == 0:
                train_result = train(agents, replay_buffer)
                if train_result is not None:
                    last_actor_loss, last_critic_loss, actor_gn, critic_gn, q_val = train_result
                    last_q_value = q_val
                    actor_grad_norms.append(actor_gn)
                    critic_grad_norms.append(critic_gn)
                    q_values.append(q_val)
                
                # Update target networks
                for agent in agents:
                    agent.update_targets()

        # Get episode metrics
        metrics = env.get_metrics()
        episode_metrics.append(metrics)
        
        avg_episode_reward = current_episode_reward / STEPS_PER_EPISODE
        episode_rewards.append(avg_episode_reward)
        
        # Track best performance
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            print(f"🌟 New best reward: {best_reward:.4f}")
        
        # Log theo từng episode với thông tin đầy đủ
        episode_time = time.time() - episode_start_time
        avg_reward_10 = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_rewards[-1]
        avg_reward_50 = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_rewards[-1]
        
        # Get current learning rate and epsilon
        current_lr_actor = agents[0].actor_optimizer.param_groups[0]['lr']
        current_epsilon = agents[0].epsilon
        current_expert_eps = agents[0].expert_epsilon
        
        if last_actor_loss is not None and last_critic_loss is not None:
            print(f"Ep {episode + 1:3d}/{NUM_EPISODES} | R: {episode_rewards[-1]:+.4f} | R(10): {avg_reward_10:+.4f} | R(50): {avg_reward_50:+.4f} | Succ: {metrics['success_rate']:.3f} | Thr: {metrics['avg_throughput']:.3f} | ALoss: {last_actor_loss:.4f} | CLoss: {last_critic_loss:.4f} | ε: {current_epsilon:.3f} | Exp: {current_expert_eps:.3f} | Time: {episode_time:.1f}s")
        else:
            print(f"Ep {episode + 1:3d}/{NUM_EPISODES} | R: {episode_rewards[-1]:+.4f} | R(10): {avg_reward_10:+.4f} | R(50): {avg_reward_50:+.4f} | Succ: {metrics['success_rate']:.3f} | Thr: {metrics['avg_throughput']:.3f} | [Warmup] | Exp: {current_expert_eps:.3f} | Time: {episode_time:.1f}s")

    training_time = time.time() - training_start_time
    print(f"\nTraining Finished!")
    print(f"Total training time: {training_time/60:.2f} minutes ({training_time:.1f} seconds)")
    print(f"Average time per episode: {training_time/NUM_EPISODES:.1f} seconds")
    
    # Print training stability metrics
    if len(actor_grad_norms) > 0:
        print(f"\nTraining Stability Metrics:")
        print(f"  Actor Grad Norm  - Mean: {np.mean(actor_grad_norms):.4f}, Max: {np.max(actor_grad_norms):.4f}, Final: {actor_grad_norms[-1]:.4f}")
        print(f"  Critic Grad Norm - Mean: {np.mean(critic_grad_norms):.4f}, Max: {np.max(critic_grad_norms):.4f}, Final: {critic_grad_norms[-1]:.4f}")
        print(f"  Q-Value          - Mean: {np.mean(q_values):.4f}, Std: {np.std(q_values):.4f}, Final: {q_values[-1]:.4f}")
        print(f"  Final LR (Actor): {agents[0].actor_optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Final Epsilon: {agents[0].epsilon:.4f}")
    
    # --- Lưu kết quả training vào CSV với đầy đủ metrics ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'training_rewards_{timestamp}.csv'
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Episode', 'Reward', 'Success_Rate', 'Energy_Efficiency', 'Avg_Energy', 'Avg_Throughput', 'Throughput_Efficiency'])
        for i, (reward, metrics) in enumerate(zip(episode_rewards, episode_metrics), 1):
            writer.writerow([i, reward, metrics['success_rate'], metrics['energy_efficiency'], metrics['avg_energy_per_step'], metrics['avg_throughput'], metrics['throughput_efficiency']])
    print(f"✓ Đã lưu training rewards và metrics vào: {csv_filename}")
    
    # --- Lưu model checkpoint ---
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        checkpoint = {
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'target_actor_state_dict': agent.target_actor.state_dict(),
            'target_critic_state_dict': agent.target_critic.state_dict(),
        }
        checkpoint_path = f'{checkpoint_dir}/agent_{i}_{timestamp}.pth'
        torch.save(checkpoint, checkpoint_path)
    print(f"✓ Đã lưu model checkpoints vào thư mục: {checkpoint_dir}/")

    # --- Bắt đầu phần đánh giá Baselines ---
    
    # Lấy các tham số từ môi trường đã tạo
    env_params = {
        'pga_gain': env.pga_gain,
        'jammer_power': env.jammer_power,
        'noise_power': env.noise_power,
    }
    # Khởi tạo các baselines
    dt_agent = DirectTransmission(NUM_AGENTS, env.action_dim)
    greedy_agent = GreedyStrategy(NUM_AGENTS, env.action_dim, env_params)
    fh_agent = FrequencyHopping(NUM_AGENTS, env_params)

    # Đánh giá baselines với đầy đủ metrics
    num_eval_episodes = 50  # Reduced from 200 for faster evaluation
    dt_metrics = evaluate_baseline(dt_agent, env, num_eval_episodes)
    greedy_metrics = evaluate_baseline(greedy_agent, env, num_eval_episodes)
    fh_metrics = evaluate_baseline(fh_agent, env, num_eval_episodes)
    
    # MADDPG metrics từ last 50 episodes
    maddpg_last_n = 50
    maddpg_metrics = {
        'reward': np.mean(episode_rewards[-maddpg_last_n:]),
        'success_rate': np.mean([m['success_rate'] for m in episode_metrics[-maddpg_last_n:]]),
        'energy_efficiency': np.mean([m['energy_efficiency'] for m in episode_metrics[-maddpg_last_n:]]),
        'avg_energy': np.mean([m['avg_energy_per_step'] for m in episode_metrics[-maddpg_last_n:]]),
        'avg_throughput': np.mean([m['avg_throughput'] for m in episode_metrics[-maddpg_last_n:]]),
        'throughput_efficiency': np.mean([m['throughput_efficiency'] for m in episode_metrics[-maddpg_last_n:]])
    }

    # In ra bảng so sánh cuối cùng với đầy đủ metrics
    print("\n" + "="*80)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Method':<25} {'Reward':>10} {'Success':>8} {'Throughput':>11} {'Energy Eff':>10} {'Thr Eff':>9}")
    print("-"*80)
    print(f"{'Direct Transmission':<25} {dt_metrics['reward']:>10.4f} {dt_metrics['success_rate']:>8.3f} {dt_metrics['avg_throughput']:>11.4f} {dt_metrics['energy_efficiency']:>10.4f} {dt_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'Greedy Strategy':<25} {greedy_metrics['reward']:>10.4f} {greedy_metrics['success_rate']:>8.3f} {greedy_metrics['avg_throughput']:>11.4f} {greedy_metrics['energy_efficiency']:>10.4f} {greedy_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'Frequency Hopping':<25} {fh_metrics['reward']:>10.4f} {fh_metrics['success_rate']:>8.3f} {fh_metrics['avg_throughput']:>11.4f} {fh_metrics['energy_efficiency']:>10.4f} {fh_metrics['throughput_efficiency']:>9.4f}")
    print(f"{'MADDPG + JM (Ours)':<25} {maddpg_metrics['reward']:>10.4f} {maddpg_metrics['success_rate']:>8.3f} {maddpg_metrics['avg_throughput']:>11.4f} {maddpg_metrics['energy_efficiency']:>10.4f} {maddpg_metrics['throughput_efficiency']:>9.4f}")
    print("="*80)
    
    # Calculate improvement percentages
    best_baseline_reward = max(dt_metrics['reward'], greedy_metrics['reward'], fh_metrics['reward'])
    best_baseline_throughput = max(dt_metrics['avg_throughput'], greedy_metrics['avg_throughput'], fh_metrics['avg_throughput'])
    best_baseline_success = max(dt_metrics['success_rate'], greedy_metrics['success_rate'], fh_metrics['success_rate'])
    
    reward_improvement = ((maddpg_metrics['reward'] - best_baseline_reward) / abs(best_baseline_reward)) * 100
    throughput_improvement = ((maddpg_metrics['avg_throughput'] - best_baseline_throughput) / abs(best_baseline_throughput)) * 100
    success_improvement = ((maddpg_metrics['success_rate'] - best_baseline_success) / abs(best_baseline_success)) * 100
    
    print(f"\n🎯 MADDPG vs Best Baseline (Greedy: {greedy_metrics['reward']:.4f}):")
    print(f"   Reward:      {maddpg_metrics['reward']:>7.4f} vs {best_baseline_reward:>7.4f} = {reward_improvement:>+7.2f}%")
    print(f"   Success Rate:{maddpg_metrics['success_rate']:>7.3f} vs {best_baseline_success:>7.3f} = {success_improvement:>+7.2f}%")
    print(f"   Throughput:  {maddpg_metrics['avg_throughput']:>7.4f} vs {best_baseline_throughput:>7.4f} = {throughput_improvement:>+7.2f}%")
    
    if maddpg_metrics['reward'] > best_baseline_reward:
        print(f"\n✅ SUCCESS! MADDPG BEATS ALL BASELINES!")
    else:
        gap = best_baseline_reward - maddpg_metrics['reward']
        print(f"\n⚠️  MADDPG needs {gap:.4f} more reward to beat baselines")
    
    # Lưu kết quả baseline vào CSV với đầy đủ metrics
    baseline_filename = f'baseline_comparison_{timestamp}.csv'
    with open(baseline_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'Average_Reward', 'Success_Rate', 'Energy_Efficiency', 'Avg_Energy', 'Avg_Throughput', 'Throughput_Efficiency'])
        writer.writerow(['Direct Transmission', dt_metrics['reward'], dt_metrics['success_rate'], dt_metrics['energy_efficiency'], dt_metrics['avg_energy'], dt_metrics['avg_throughput'], dt_metrics['throughput_efficiency']])
        writer.writerow(['Greedy Strategy', greedy_metrics['reward'], greedy_metrics['success_rate'], greedy_metrics['energy_efficiency'], greedy_metrics['avg_energy'], greedy_metrics['avg_throughput'], greedy_metrics['throughput_efficiency']])
        writer.writerow(['Frequency Hopping', fh_metrics['reward'], fh_metrics['success_rate'], fh_metrics['energy_efficiency'], fh_metrics['avg_energy'], fh_metrics['avg_throughput'], fh_metrics['throughput_efficiency']])
        writer.writerow(['MADDPG + JM', maddpg_metrics['reward'], maddpg_metrics['success_rate'], maddpg_metrics['energy_efficiency'], maddpg_metrics['avg_energy'], maddpg_metrics['avg_throughput'], maddpg_metrics['throughput_efficiency']])
    print(f"✓ Đã lưu kết quả baseline vào: {baseline_filename}")
    
    # Vẽ đồ thị phần thưởng của MADDPG với enhanced visualization including throughput
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    
    # Plot 1: Rewards với moving average
    window = 10
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 0].plot(range(window-1, len(episode_rewards)), smoothed_rewards, label=f'MADDPG (MA-{window})', linewidth=2, color='blue')
    axes[0, 0].plot(episode_rewards, alpha=0.3, linewidth=0.8, color='blue', label='MADDPG (Raw)')
    axes[0, 0].axhline(y=dt_metrics['reward'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['reward']:.4f}")
    axes[0, 0].axhline(y=greedy_metrics['reward'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['reward']:.4f}")
    axes[0, 0].axhline(y=fh_metrics['reward'], color='m', linestyle='--', linewidth=2, label=f"FH: {fh_metrics['reward']:.4f}")
    axes[0, 0].set_xlabel("Episode", fontsize=12)
    axes[0, 0].set_ylabel("Average Reward per Step", fontsize=12)
    axes[0, 0].set_title("MADDPG Training vs. Baselines - Rewards", fontsize=14, fontweight='bold')
    axes[0, 0].legend(loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Success Rate with baselines
    success_rates = [m['success_rate'] for m in episode_metrics]
    if len(success_rates) >= window:
        smoothed_success = np.convolve(success_rates, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(range(window-1, len(success_rates)), smoothed_success, color='orange', linewidth=2, label='MADDPG')
    else:
        axes[0, 1].plot(success_rates, color='orange', linewidth=2, label='MADDPG')
    axes[0, 1].axhline(y=dt_metrics['success_rate'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['success_rate']:.3f}")
    axes[0, 1].axhline(y=greedy_metrics['success_rate'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['success_rate']:.3f}")
    axes[0, 1].axhline(y=fh_metrics['success_rate'], color='m', linestyle='--', linewidth=2, label=f"FH: {fh_metrics['success_rate']:.3f}")
    axes[0, 1].set_xlabel("Episode", fontsize=12)
    axes[0, 1].set_ylabel("Success Rate", fontsize=12)
    axes[0, 1].set_title("Transmission Success Rate Over Time", fontsize=14, fontweight='bold')
    axes[0, 1].legend(loc='best')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Energy Efficiency with baselines
    energy_effs = [m['energy_efficiency'] for m in episode_metrics]
    if len(energy_effs) >= window:
        smoothed_ee = np.convolve(energy_effs, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(range(window-1, len(energy_effs)), smoothed_ee, color='green', linewidth=2, label='MADDPG')
    else:
        axes[1, 0].plot(energy_effs, color='green', linewidth=2, label='MADDPG')
    axes[1, 0].axhline(y=dt_metrics['energy_efficiency'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['energy_efficiency']:.4f}")
    axes[1, 0].axhline(y=greedy_metrics['energy_efficiency'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['energy_efficiency']:.4f}")
    axes[1, 0].axhline(y=fh_metrics['energy_efficiency'], color='m', linestyle='--', linewidth=2, label=f"FH: {fh_metrics['energy_efficiency']:.4f}")
    axes[1, 0].set_xlabel("Episode", fontsize=12)
    axes[1, 0].set_ylabel("Energy Efficiency (Success/Energy)", fontsize=12)
    axes[1, 0].set_title("Energy Efficiency Over Time", fontsize=14, fontweight='bold')
    axes[1, 0].legend(loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Average Throughput Over Time
    throughputs = [m['avg_throughput'] for m in episode_metrics]
    if len(throughputs) >= window:
        smoothed_throughput = np.convolve(throughputs, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(throughputs)), smoothed_throughput, color='purple', linewidth=2, label='MADDPG')
    else:
        axes[1, 1].plot(throughputs, color='purple', linewidth=2, label='MADDPG')
    axes[1, 1].axhline(y=dt_metrics['avg_throughput'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['avg_throughput']:.4f}")
    axes[1, 1].axhline(y=greedy_metrics['avg_throughput'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['avg_throughput']:.4f}")
    axes[1, 1].axhline(y=fh_metrics['avg_throughput'], color='m', linestyle='--', linewidth=2, label=f"FH: {fh_metrics['avg_throughput']:.4f}")
    axes[1, 1].set_xlabel("Episode", fontsize=12)
    axes[1, 1].set_ylabel("Average Throughput (bits/s/Hz)", fontsize=12)
    axes[1, 1].set_title("Throughput Comparison Over Time", fontsize=14, fontweight='bold')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 5: Throughput Efficiency Over Time
    throughput_effs = [m['throughput_efficiency'] for m in episode_metrics]
    if len(throughput_effs) >= window:
        smoothed_thr_eff = np.convolve(throughput_effs, np.ones(window)/window, mode='valid')
        axes[0, 2].plot(range(window-1, len(throughput_effs)), smoothed_thr_eff, color='brown', linewidth=2, label='MADDPG')
    else:
        axes[0, 2].plot(throughput_effs, color='brown', linewidth=2, label='MADDPG')
    axes[0, 2].axhline(y=dt_metrics['throughput_efficiency'], color='r', linestyle='--', linewidth=2, label=f"DT: {dt_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].axhline(y=greedy_metrics['throughput_efficiency'], color='g', linestyle='--', linewidth=2, label=f"Greedy: {greedy_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].axhline(y=fh_metrics['throughput_efficiency'], color='m', linestyle='--', linewidth=2, label=f"FH: {fh_metrics['throughput_efficiency']:.4f}")
    axes[0, 2].set_xlabel("Episode", fontsize=12)
    axes[0, 2].set_ylabel("Throughput Efficiency (bits/J)", fontsize=12)
    axes[0, 2].set_title("Throughput per Energy Over Time", fontsize=14, fontweight='bold')
    axes[0, 2].legend(loc='best')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 6: Comparative Bar Chart
    methods = ['DT', 'Greedy', 'FH', 'MADDPG']
    rewards = [dt_metrics['reward'], greedy_metrics['reward'], fh_metrics['reward'], maddpg_metrics['reward']]
    colors = ['red', 'green', 'magenta', 'blue']
    bars = axes[1, 2].bar(methods, rewards, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    # Highlight the best
    best_idx = np.argmax(rewards)
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(4)
    axes[1, 2].set_ylabel("Average Reward", fontsize=12)
    axes[1, 2].set_title("Final Performance Comparison", fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, rewards)):
        height = bar.get_height()
        axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    results_plot_filename = f'training_results_with_throughput_{timestamp}.png'
    plt.savefig(results_plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Đã lưu đồ thị toàn diện với throughput vào file: {results_plot_filename}")
    plt.show()