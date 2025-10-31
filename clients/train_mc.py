import os
import pickle
from collections import defaultdict
import random
import numpy as np
from typing import List, Tuple, Optional
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.robot_env import GridWorldEnv

def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def encode_visited(wp_list, visited_set):
    code = 0
    for i, wp in enumerate(wp_list):
        if wp in visited_set:
            code |= (1 << i)
    return code

# --- THAM SỐ CHO MC ES ---
gamma = 0.99
num_episodes = 20000
max_steps_per_episode = 1000 

actions = ['up', 'right', 'down', 'left']
num_actions = len(actions)

# --- KHỞI TẠO THEO MC ES ---
mc_Q = defaultdict(lambda: {a: 0.0 for a in actions})
mc_Returns = defaultdict(list)
mc_policy = defaultdict(lambda: random.choice(actions))


models_dir = os.path.join(os.path.dirname(__file__), "models") if '__file__' in globals() else "./models"
os.makedirs(models_dir, exist_ok=True)

MC_QFILE = os.path.join(models_dir, "mc_qtable.pkl")
MC_RETURNS_FILE = os.path.join(models_dir, "mc_returns.pkl")
MC_POLICY_FILE = os.path.join(models_dir, "mc_policy.pkl")

if os.path.exists(MC_QFILE):
    try:
        with open(MC_QFILE, 'rb') as f:
            mc_Q.update(pickle.load(f))
        print(f"Đã tải mc_Q từ {MC_QFILE}")
    except Exception as e:
        print(f"Lỗi khi tải mc_Q: {e}. Bắt đầu với Q trống.")
        mc_Q = defaultdict(lambda: {a: 0.0 for a in actions})

if os.path.exists(MC_RETURNS_FILE):
    try:
        with open(MC_RETURNS_FILE, 'rb') as f:
            mc_Returns.update(pickle.load(f))
        print(f"Đã tải mc_Returns từ {MC_RETURNS_FILE}")
    except Exception as e:
        print(f"Lỗi khi tải mc_Returns: {e}. Bắt đầu với Returns trống.")
        mc_Returns = defaultdict(list)

if os.path.exists(MC_POLICY_FILE):
    try:
        with open(MC_POLICY_FILE, 'rb') as f:
            loaded_policy_dict = pickle.load(f)
        mc_policy.update(loaded_policy_dict)
        print(f"Đã tải mc_policy từ {MC_POLICY_FILE}")
    except Exception as e:
        print(f"Lỗi khi tải mc_policy: {e}. Bắt đầu với policy ngẫu nhiên.")
        mc_policy = defaultdict(lambda: random.choice(actions))

# Map cố định
width, height = 10, 8
start = (0, 0)
goal = (9, 7)
obstacles = [(1,1), (2,3), (4,4), (5,1), (7,6)]
waypoints = [(3,2), (6,5)]

env = GridWorldEnv(width, height, start, goal, obstacles, waypoints, max_steps=max_steps_per_episode)

# Điều chỉnh reward
env.step_penalty = -2.0
env.revisit_penalty = -3.0
env.waypoint_reward = 30.0
env.goal_reward = 100.0
env.goal_before_waypoints_penalty = -10.0

total_rewards = []


# --- BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN CHÍNH ---
for episode in range(num_episodes): # 
    env.reset(start=start, goal=goal, obstacles=obstacles, waypoints=waypoints)

    # <<< Triển khai Exploring Starts (ES) đầy đủ >>>
    # 1. Chọn S0 ngẫu nhiên
    all_cells = [(x, y) for x in range(width) for y in range(height) if (x, y) not in env.obstacles]
   
    valid_start_cells = [cell for cell in all_cells if cell != start and cell != goal]
    if not valid_start_cells: # Phòng trường hợp grid quá nhỏ/chật
        valid_start_cells = all_cells if all_cells else [(0,0)] # Fallback
    env.state = random.choice(valid_start_cells)
    # <<< HẾT SỬA LỖI NHỎ >>>
    num_visited = random.randint(0, len(waypoints))
    env.visited_waypoints = set(random.sample(waypoints, num_visited))

    # Lấy trạng thái S0
    state_xy = env.get_state()
    dist_to_next = min([manhattan_distance(state_xy, wp) for wp in waypoints if wp not in env.visited_waypoints] +
                       [manhattan_distance(state_xy, goal)] if len(env.visited_waypoints) == len(waypoints) else [float('inf')])
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    full_state = (state_xy[0], state_xy[1], visited_code, dist_to_next)

    # 2. Chọn A0 ngẫu nhiên
    action_name = random.choice(actions)

    trajectory = []

    # Thực hiện bước 
    action_idx = actions.index(action_name)
    next_state_xy, reward, done, _ = env.step(action_idx)
    trajectory.append((full_state, action_name, reward))

    # Cập nhật state thành S1
    dist_to_next = min([manhattan_distance(next_state_xy, wp) for wp in waypoints if wp not in env.visited_waypoints] +
                         [manhattan_distance(next_state_xy, goal)] if len(env.visited_waypoints) == len(waypoints) else [float('inf')])
    visited_code = encode_visited(env.waypoints, env.visited_waypoints)
    full_state = (next_state_xy[0], next_state_xy[1], visited_code, dist_to_next)

    episode_reward = reward
    steps = 1

    # <<< Generate episode following pi (tham lam) >>>
    while not done and steps < max_steps_per_episode:
        action_name = mc_policy[full_state]
        action_idx = actions.index(action_name)
        next_state_xy, reward, done, _ = env.step(action_idx)
        trajectory.append((full_state, action_name, reward))
        dist_to_next = min([manhattan_distance(next_state_xy, wp) for wp in waypoints if wp not in env.visited_waypoints] +
                             [manhattan_distance(next_state_xy, goal)] if len(env.visited_waypoints) == len(waypoints) else [float('inf')])
        visited_code = encode_visited(env.waypoints, env.visited_waypoints)
        full_state = (next_state_xy[0], next_state_xy[1], visited_code, dist_to_next)
        episode_reward += reward
        steps += 1

    # Cập nhật Q-value và Policy (theo logic MC ES)
    G = 0
    visited_state_actions = set()
    for t in reversed(range(len(trajectory))):
        state, action, r = trajectory[t]
        G = r + gamma * G
        state_action = (state, action)

        if state_action not in visited_state_actions:
            visited_state_actions.add(state_action)
            mc_Returns[state_action].append(G)
            mc_Q[state][action] = np.mean(mc_Returns[state_action])
            best_action = max(mc_Q[state], key=mc_Q[state].get)
            mc_policy[state] = best_action

    total_rewards.append(episode_reward)
    if (episode + 1) % 1000 == 0:
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f} - Steps: {steps}")

# --- PHẦN LƯU FILE (Giữ nguyên) ---
# ... (code lưu mc_Q, mc_Returns, mc_policy) ...
models_dir = os.path.join(os.path.dirname(__file__), "models") if '__file__' in globals() else "./models"
os.makedirs(models_dir, exist_ok=True)

MC_QFILE = os.path.join(models_dir, "mc_qtable.pkl")
MC_RETURNS_FILE = os.path.join(models_dir, "mc_returns.pkl")
MC_POLICY_FILE = os.path.join(models_dir, "mc_policy.pkl")

with open(MC_QFILE, 'wb') as f:
    pickle.dump(dict(mc_Q), f)
print(f"Huấn luyện hoàn tất. Q-table được lưu tại: {MC_QFILE}")
<<<<<<< Updated upstream
=======

with open(MC_RETURNS_FILE, 'wb') as f:
    pickle.dump(dict(mc_Returns), f)
print(f"Lịch sử Returns được lưu tại: {MC_RETURNS_FILE}")

with open(MC_POLICY_FILE, 'wb') as f:
    pickle.dump(dict(mc_policy), f)
print(f"Chính sách Policy được lưu tại: {MC_POLICY_FILE}")
>>>>>>> Stashed changes
