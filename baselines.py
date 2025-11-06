# baselines.py

import numpy as np

class DirectTransmission:
    """
    Chiến lược cơ sở: Luôn cố gắng truyền tin một cách ngây thơ.
    Nó không quan tâm đến trạng thái của môi trường.
    Hành động: Luôn gửi bit '1' (khuếch đại) và dùng chế độ D2D.
    """
    def __init__(self, num_agents, action_dim):
        self.num_agents = num_agents
        self.action_dim = action_dim
        # Hành động cố định: [pga_choice=1, mode_choice=-1 (D2D)]
        self.fixed_action = np.array([1.0, -1.0])

    def select_actions(self, states):
        # Trả về cùng một hành động cho tất cả các agent, bỏ qua trạng thái
        return [self.fixed_action for _ in range(self.num_agents)]

class GreedyStrategy:
    """
    Chiến lược tham lam: Tại mỗi bước, chọn hành động (chế độ D2D hoặc Relay)
    tối đa hóa SINR tức thời.
    """
    def __init__(self, num_agents, action_dim, env_params):
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.pga_gain = env_params['pga_gain']
        self.jammer_power = env_params['jammer_power']
        self.noise_power = env_params['noise_power']

    def select_actions(self, states):
        actions = []
        for i in range(self.num_agents):
            state = states[i]
            
            # Khôi phục các giá trị hệ số kênh từ trạng thái (đã được log10)
            g_su_du = 10**state[1]
            g_su_rbs = 10**state[2]
            g_rbs_du = 10**state[3]
            g_jam_su = 10**state[4]
            g_jam_du = 10**state[5]
            g_jam_rbs = 10**state[6]

            # --- Tính toán SINR cho chế độ D2D ---
            # Greedy luôn chọn gửi bit '1' (khuếch đại)
            pga_choice = self.pga_gain
            num_d2d = g_jam_su * g_su_du * (pga_choice**2) * self.jammer_power
            den_d2d = g_jam_du * self.jammer_power + self.noise_power
            sinr_d2d = num_d2d / (den_d2d + 1e-9)

            # --- Tính toán SINR cho chế độ Relay ---
            # Chặng 1: SU -> rBS
            num1_relay = g_jam_su * g_su_rbs * (pga_choice**2) * self.jammer_power
            den1_relay = g_jam_rbs * self.jammer_power + self.noise_power
            sinr1_relay = num1_relay / (den1_relay + 1e-9)
            
            # Chặng 2: rBS -> DU
            num2_relay = g_jam_rbs * g_rbs_du * (pga_choice**2) * self.jammer_power
            den2_relay = g_jam_du * self.jammer_power + self.noise_power
            sinr2_relay = num2_relay / (den2_relay + 1e-9)
            
            sinr_relay = min(sinr1_relay, sinr2_relay)

            # --- So sánh và đưa ra quyết định ---
            if sinr_d2d >= sinr_relay:
                # Chọn D2D
                action = np.array([1.0, -1.0]) 
            else:
                # Chọn Relay
                action = np.array([1.0, 1.0])
            
            actions.append(action)
        return actions

class FrequencyHopping:
    """
    Chiến lược kinh điển: Nhảy tần để tránh nhiễu.
    Mô hình này không dùng Jamming Modulation, nên ta cần tính toán reward một cách riêng biệt.
    """
    def __init__(self, num_agents, env_params):
        self.num_agents = num_agents
        # Giả định có 40 kênh tần số, và jammer chiếm 10 kênh trong đó
        self.num_channels = 40
        self.num_jammed_channels = 10
        self.p_collision = self.num_jammed_channels / self.num_channels

        # Các tham số cho truyền thông bình thường (khi không bị nhiễu)
        self.tx_power = 0.1 # Công suất phát thông thường (Watt)
        self.noise_power = env_params['noise_power']

    def get_rewards(self, states):
        """
        Tính toán phần thưởng kỳ vọng cho chiến lược FH.
        Không có 'hành động' cụ thể, kết quả mang tính xác suất.
        """
        rewards = []
        for i in range(self.num_agents):
            state = states[i]
            # Lấy hệ số kênh SU-DU
            g_su_du = 10**state[1]

            # Khi hop thành công (không bị nhiễu)
            sinr_success = (self.tx_power * g_su_du) / (self.noise_power + 1e-9)
            throughput_success = np.log2(1 + sinr_success)

            # Khi hop thất bại (rơi vào kênh nhiễu)
            throughput_fail = 0

            # Phần thưởng kỳ vọng
            expected_reward = (1 - self.p_collision) * throughput_success + self.p_collision * throughput_fail
            rewards.append(expected_reward)
        
        return rewards