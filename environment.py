# environment.py

import numpy as np

class CommEnvironment:
    def __init__(self, num_agents=5, area_size=500, jammer_power=1.0, noise_power=1e-4, path_loss_exponent=2.0):
        print("Initializing Communication Environment...")
        self.num_agents = num_agents
        self.area_size = area_size
        self.jammer_power = jammer_power  # P_J
        self.noise_power = noise_power    # sigma_R^2
        self.path_loss_exponent = path_loss_exponent # alpha

        # Các thực thể: SU, DU, Jammer, Relay Base Station (rBS)
        self.sus = None
        self.dus = None
        self.jammer = None
        self.rbs = np.array([area_size / 2, area_size / 2]) # rBS đặt ở trung tâm

        # Các tham số của Jamming Modulation (JM)
        self.pga_gain = 100  # Hệ số khuếch đại 'a' khi gửi bit '1'

        self.state_dim = 8 # Kích thước vector trạng thái cho mỗi agent
        self.action_dim = 2 # Kích thước vector hành động cho mỗi agent
        
        # Metrics tracking
        self.total_steps = 0
        self.successful_transmissions = 0
        self.total_energy = 0
        self.total_throughput = 0  # Track cumulative throughput
        self.sinr_threshold = 0.15  # Standard threshold

        self.reset()
        print("Environment Initialized.")

    def reset(self):
        """Khởi tạo lại vị trí của các thực thể cho mỗi episode mới."""
        self.sus = np.random.rand(self.num_agents, 2) * self.area_size
        self.dus = np.random.rand(self.num_agents, 2) * self.area_size
        self.jammer = np.random.rand(1, 2) * self.area_size
        
        # Reset metrics
        self.total_steps = 0
        self.successful_transmissions = 0
        self.total_energy = 0
        self.total_throughput = 0
        
        # Trả về trạng thái ban đầu của tất cả các agent
        return self._get_states()
    
    def get_metrics(self):
        """Return current episode metrics"""
        if self.total_steps == 0:
            return {
                'success_rate': 0.0,
                'energy_efficiency': 0.0,
                'avg_energy_per_step': 0.0,
                'avg_throughput': 0.0,
                'throughput_efficiency': 0.0
            }
        return {
            'success_rate': self.successful_transmissions / (self.total_steps * self.num_agents),
            'energy_efficiency': self.successful_transmissions / max(self.total_energy, 1e-9),
            'avg_energy_per_step': self.total_energy / self.total_steps,
            'avg_throughput': self.total_throughput / (self.total_steps * self.num_agents),
            'throughput_efficiency': self.total_throughput / max(self.total_energy, 1e-9)
        }

    def _get_distance(self, pos1, pos2):
        return np.sqrt(np.sum((pos1 - pos2)**2))

    def _get_channel_gain(self, pos1, pos2):
        dist = self._get_distance(pos1, pos2)
        # Để tránh chia cho 0, thêm một epsilon nhỏ
        return (dist + 1e-6) ** -self.path_loss_exponent

    def _get_states(self):
        """Tạo vector trạng thái cho tất cả các agent."""
        states = []
        jnr = self.jammer_power / self.noise_power

        for i in range(self.num_agents):
            su_pos = self.sus[i]
            du_pos = self.dus[i]
            jammer_pos = self.jammer[0]

            # Các hệ số kênh liên quan đến agent i
            g_su_du = self._get_channel_gain(su_pos, du_pos)
            g_su_rbs = self._get_channel_gain(su_pos, self.rbs)
            g_rbs_du = self._get_channel_gain(self.rbs, du_pos)
            
            # Các hệ số kênh liên quan đến jammer
            g_jam_su = self._get_channel_gain(jammer_pos, su_pos)  # h1^2
            g_jam_du = self._get_channel_gain(jammer_pos, du_pos)  # h3^2
            g_jam_rbs = self._get_channel_gain(jammer_pos, self.rbs)
            
            # Vector trạng thái: JNR và các hệ số kênh quan trọng
            state = [
                np.log10(jnr + 1e-9),
                np.log10(g_su_du + 1e-9),
                np.log10(g_su_rbs + 1e-9),
                np.log10(g_rbs_du + 1e-9),
                np.log10(g_jam_su + 1e-9),
                np.log10(g_jam_du + 1e-9),
                np.log10(g_jam_rbs + 1e-9),
                self._get_distance(su_pos, du_pos) / self.area_size # Khoảng cách chuẩn hóa
            ]
            states.append(np.array(state, dtype=np.float32))
        return states

    def step(self, actions):
        """
        Thực thi hành động của tất cả các agent và trả về trạng thái mới, phần thưởng.
        `actions` là một list, mỗi phần tử là hành động của một agent.
        Hành động của mỗi agent là một vector 2 chiều:
        - action[0]: Lựa chọn hệ số khuếch đại (ví dụ: > 0 là gửi '1', <= 0 là gửi '0')
        - action[1]: Lựa chọn chế độ giao tiếp (> 0 là Relay, <= 0 là D2D)
        """
        rewards = []
        info = {'sinr_values': [], 'modes': [], 'pga_choices': []}
        
        # Di chuyển các thực thể một cách ngẫu nhiên
        self.sus += np.random.randn(self.num_agents, 2) * 5.0 # Tốc độ di chuyển
        self.dus += np.random.randn(self.num_agents, 2) * 5.0
        self.jammer += np.random.randn(1, 2) * 10.0 # UAV di chuyển nhanh hơn

        # Giữ các thực thể trong vùng mô phỏng
        self.sus = np.clip(self.sus, 0, self.area_size)
        self.dus = np.clip(self.dus, 0, self.area_size)
        self.jammer = np.clip(self.jammer, 0, self.area_size)
        
        self.total_steps += 1

        for i in range(self.num_agents):
            action = actions[i]
            su_pos = self.sus[i]
            du_pos = self.dus[i]
            jammer_pos = self.jammer[0]

            # Giải mã hành động
            #pga_choice = self.pga_gain if action[0] > 0 else 0  # Gửi bit '1' hay '0'
            pga_choice = ((action[0] + 1) / 2) * self.pga_gain  # Chuẩn hóa từ [0,1] sang [0, pga_gain]
            mode_choice = 'relay' if action[1] > 0 else 'd2d'
            
            # Track energy consumption
            energy_cost = (pga_choice / self.pga_gain) ** 2  # Normalized energy
            if mode_choice == 'relay':
                energy_cost *= 1.5  # Relay costs more energy
            self.total_energy += energy_cost
            
            # Tính toán SINR và phần thưởng dựa trên hành động
            h1_sq = self._get_channel_gain(jammer_pos, su_pos)
            h3_sq = self._get_channel_gain(jammer_pos, du_pos)
            
            sinr = 0
            if mode_choice == 'd2d':
                h2_sq = self._get_channel_gain(su_pos, du_pos)
                numerator = h1_sq * h2_sq * (pga_choice**2) * self.jammer_power
                denominator = h3_sq * self.jammer_power + self.noise_power
                sinr = numerator / (denominator + 1e-9)

            elif mode_choice == 'relay':
                # Giả định đơn giản: SINR của kết nối qua relay là min của hai chặng
                # Chặng 1: SU -> rBS
                h_jam_rbs_sq = self._get_channel_gain(jammer_pos, self.rbs)
                h_su_rbs_sq = self._get_channel_gain(su_pos, self.rbs)
                num1 = h1_sq * h_su_rbs_sq * (pga_choice**2) * self.jammer_power
                den1 = h_jam_rbs_sq * self.jammer_power + self.noise_power
                sinr1 = num1 / (den1 + 1e-9)

                # Chặng 2: rBS -> DU
                # Giả sử rBS cũng dùng JM với hệ số khuếch đại tương tự
                h1_relay_sq = self._get_channel_gain(jammer_pos, self.rbs)
                h2_relay_sq = self._get_channel_gain(self.rbs, du_pos)
                h3_relay_sq = self._get_channel_gain(jammer_pos, du_pos)
                num2 = h1_relay_sq * h2_relay_sq * (pga_choice**2) * self.jammer_power
                den2 = h3_relay_sq * self.jammer_power + self.noise_power
                sinr2 = num2 / (den2 + 1e-9)
                
                sinr = min(sinr1, sinr2)
            
            # Track success
            is_successful = sinr > self.sinr_threshold
            if is_successful:
                self.successful_transmissions += 1

            # Balanced reward function (không quá cao để baselines hưởng lợi)
            
            # Calculate throughput (bits/s/Hz)
            throughput = np.log2(1 + sinr)
            self.total_throughput += throughput
            
            if is_successful:
                # Success reward - moderate
                base_reward = 1.0
                throughput_reward = throughput
                
                # Small mode bonus
                if mode_choice == 'd2d':
                    mode_bonus = 0.1
                else:
                    mode_bonus = 0.05
                
                # Energy efficiency
                if energy_cost > 0:
                    energy_bonus = 0.1 / (1.0 + energy_cost)
                else:
                    energy_bonus = 0.1
                
                reward = base_reward + throughput_reward + mode_bonus + energy_bonus
                
            else:
                # Failure - small guidance
                sinr_progress = 0.3 * (sinr / self.sinr_threshold)
                energy_penalty = 0.05 * energy_cost
                
                reward = sinr_progress - energy_penalty
            
            rewards.append(reward)
            
            # Track info for analysis
            info['sinr_values'].append(sinr)
            info['modes'].append(mode_choice)
            info['pga_choices'].append(pga_choice)
            
        next_states = self._get_states()
        dones = [False] * self.num_agents # Giả định episode không bao giờ kết thúc sớm

        return next_states, rewards, dones