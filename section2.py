import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium import spaces
from stable_baselines3 import PPO
import os

def generate_6g_from_real_5g(filename="5g_energy_data.csv", output_file="6g_scaled_data.csv"):
    if not os.path.exists(filename):
        print(f"⚠️ Error: '{filename}' not found.")
        return None

    print(f"✅ Processing Real Data: {filename}...")
    df = pd.read_csv(filename)
    
    real_energy_5g = df['Energy'].values
    real_load = df['load'].values
    
    power_6g_active = real_energy_5g * 3.5 
    power_6g_sleep = real_energy_5g * 0.2
    
    t = np.linspace(0, len(df), len(df))
    grid_carbon = 300 + 150 * np.sin(t * (2 * np.pi / 24) + np.pi)
    
    df_6g = pd.DataFrame({
        'traffic_load': real_load,
        'power_5g_measured': real_energy_5g,
        'power_6g_active_est': power_6g_active,
        'power_6g_sleep_est': power_6g_sleep,
        'grid_carbon_intensity': grid_carbon
    })
    
    df_6g.to_csv(output_file, index=False)
    print(f"   -> Scaled 6G Dataset saved to: {output_file}")
    return df_6g

class SixG_RealData_Env(gym.Env):
    def __init__(self, df):
        super(SixG_RealData_Env, self).__init__()
        self.df = df
        self.max_idx = len(df) - 1
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(2,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None):
        self.curr_step = np.random.randint(0, max(1, self.max_idx - 2000))
        row = self.df.iloc[self.curr_step]
        return np.array([row['traffic_load'], row['grid_carbon_intensity']], dtype=np.float32), {}

    def step(self, action):
        self.curr_step += 1
        if self.curr_step >= self.max_idx:
            return np.zeros(2, dtype=np.float32), 0, True, False, {}

        row = self.df.iloc[self.curr_step]
        real_load = row['traffic_load']
        real_carbon_int = row['grid_carbon_intensity']
        
        p_sleep = row['power_6g_sleep_est']
        p_active = row['power_6g_active_est']
        
        if action == 0:   
            power_used = p_sleep
            capacity = 0.1 
        elif action == 1: 
            power_used = p_active * 0.5 
            capacity = 0.6
        else:             
            power_used = p_active 
            capacity = 1.0

        emission = (power_used / 1000.0) * real_carbon_int
        
        drops = 0
        if capacity < real_load:
            drops = (real_load - capacity)
        
        reward = -(emission * 0.05) - (drops * 50.0)

        done = (self.curr_step % 1000 == 0)
        
        next_row = self.df.iloc[self.curr_step]
        return np.array([next_row['traffic_load'], next_row['grid_carbon_intensity']], dtype=np.float32), reward, done, False, {}

def run_baseline_max_perf(df):
    total_carbon = 0
    total_drops = 0
    
    for i in range(1000):
        row = df.iloc[i]
        
        power_used = row['power_6g_active_est']
        capacity = 1.0
        
        emission = (power_used / 1000.0) * row['grid_carbon_intensity']
        total_carbon += emission
        
        if capacity < row['traffic_load']:
            total_drops += (row['traffic_load'] - capacity)
            
    return total_carbon, total_drops

def run_baseline_sleep_aware(df):
    total_carbon = 0
    total_drops = 0
    
    for i in range(1000):
        row = df.iloc[i]
        load = row['traffic_load']
        
        p_sleep = row['power_6g_sleep_est']
        p_active = row['power_6g_active_est']
        
        if load < 0.5:
            power_used = p_sleep
            capacity = 0.1
        else:
            power_used = p_active * 0.5 
            capacity = 0.6
            
        emission = (power_used / 1000.0) * row['grid_carbon_intensity']
        total_carbon += emission
        
        if capacity < load:
            total_drops += (load - capacity)
            
    return total_carbon, total_drops

df_6g = generate_6g_from_real_5g("5g_energy_data.csv")

if df_6g is not None:
    print("\n--- Training Carbon-Aware PPO ---")
    env = SixG_RealData_Env(df_6g)
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=15000)

    obs, _ = env.reset()
    env.curr_step = 0 
    
    ppo_c = 0
    ppo_d = 0
    
    for i in range(1000):
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)
        
        row = df_6g.iloc[i+1] 
        p_sleep = row['power_6g_sleep_est']
        p_active = row['power_6g_active_est']
        
        if action == 0: p = p_sleep; c = 0.1
        elif action == 1: p = p_active * 0.5; c = 0.6
        else: p = p_active; c = 1.0
        
        ppo_c += (p / 1000.0) * row['grid_carbon_intensity']
        if c < row['traffic_load']:
            ppo_d += (row['traffic_load'] - c)

    eval_slice = df_6g.iloc[0:1000]
    perf_c, perf_d = run_baseline_max_perf(eval_slice)
    sleep_c, sleep_d = run_baseline_sleep_aware(eval_slice)

    methods = ['Max-Performance\n(Kumar et al.)', 'Sleep-Aware\n(Fernando et al.)', 'DRL-PPO\n(Proposed)']
    carbons = [perf_c, sleep_c, ppo_c]
    drops = [perf_d, sleep_d, ppo_d]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    ax[0].bar(methods, carbons, color=['#e74c3c', '#27ae60', '#3498db'])
    ax[0].set_title('Total Carbon Emissions (gCO2) - Lower is Better')
    ax[0].set_ylabel('gCO2')
    ax[0].grid(axis='y', alpha=0.3)

    ax[1].bar(methods, drops, color=['#e74c3c', '#27ae60', '#3498db'])
    ax[1].set_title('QoS Violations (Dropped Traffic) - Lower is Better')
    ax[1].set_ylabel('Unserved Load')
    ax[1].grid(axis='y', alpha=0.3)

    plt.suptitle("Comparison on Real-World Scaled Data")
    plt.tight_layout()
    plt.show()

    print(f"\nFINAL RESULTS (Data-Driven):")
    print(f"1. Max-Performance: Carbon={perf_c:.0f}, Drops={perf_d:.1f}")
    print(f"2. Sleep-Aware:     Carbon={sleep_c:.0f}, Drops={sleep_d:.1f}")
    print(f"3. DRL-PPO:         Carbon={ppo_c:.0f}, Drops={ppo_d:.1f}")