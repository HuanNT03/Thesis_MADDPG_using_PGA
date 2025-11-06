#!/usr/bin/env python3
"""
Script để vẽ lại đồ thị từ file CSV đã lưu
Cách dùng: python plot_from_csv.py training_rewards_YYYYMMDD_HHMMSS.csv
"""

import sys
import csv
import matplotlib.pyplot as plt
import numpy as np
import os

def read_rewards_csv(filename):
    """Đọc rewards từ CSV"""
    rewards = []
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rewards.append(float(row['Reward']))
    return rewards

def read_baseline_csv(filename):
    """Đọc baseline results từ CSV"""
    baselines = {}
    try:
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                baselines[row['Method']] = float(row['Average_Reward'])
    except FileNotFoundError:
        print(f"⚠ Không tìm thấy file baseline: {filename}")
        return {}
    return baselines

def plot_results(rewards, baselines, output_filename='plot_from_csv.png'):
    """Vẽ đồ thị"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='MADDPG Training', linewidth=1.5, alpha=0.7)
    
    # Vẽ đường baseline
    if 'Direct Transmission' in baselines:
        plt.axhline(y=baselines['Direct Transmission'], color='r', linestyle='--', 
                   label=f"DT Baseline: {baselines['Direct Transmission']:.4f}", linewidth=2)
    if 'Greedy Strategy' in baselines:
        plt.axhline(y=baselines['Greedy Strategy'], color='g', linestyle='--', 
                   label=f"Greedy Baseline: {baselines['Greedy Strategy']:.4f}", linewidth=2)
    if 'Frequency Hopping' in baselines:
        plt.axhline(y=baselines['Frequency Hopping'], color='m', linestyle='--', 
                   label=f"FH Baseline: {baselines['Frequency Hopping']:.4f}", linewidth=2)
    
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Average Throughput per Step (log2(1+SINR))", fontsize=12)
    plt.title("MADDPG Training vs. Baselines", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Đã lưu đồ thị vào: {output_filename}")
    print(f"✓ Tổng số episodes: {len(rewards)}")
    print(f"✓ Reward trung bình cuối (50 ep): {np.mean(rewards[-50:]):.4f}")
    
    plt.show()

def find_latest_csv():
    """Tự động tìm file CSV mới nhất"""
    csv_files = [f for f in os.listdir('.') if f.startswith('training_rewards_') and f.endswith('.csv')]
    if not csv_files:
        return None
    return max(csv_files)  # Lấy file mới nhất theo tên

if __name__ == "__main__":
    # Xác định file CSV cần đọc
    if len(sys.argv) > 1:
        rewards_csv = sys.argv[1]
    else:
        # Tự động tìm file mới nhất
        rewards_csv = find_latest_csv()
        if rewards_csv is None:
            print("❌ Không tìm thấy file CSV nào!")
            print("\nHƯỚNG DẪN:")
            print("1. Chạy main.py để tạo file CSV")
            print("   hoặc")
            print("2. Chỉ định file CSV: python plot_from_csv.py training_rewards_YYYYMMDD_HHMMSS.csv")
            sys.exit(1)
        print(f"📂 Tự động chọn file mới nhất: {rewards_csv}")
    
    # Đọc dữ liệu
    if not os.path.exists(rewards_csv):
        print(f"❌ File không tồn tại: {rewards_csv}")
        sys.exit(1)
    
    rewards = read_rewards_csv(rewards_csv)
    
    # Tìm file baseline tương ứng
    timestamp = rewards_csv.replace('training_rewards_', '').replace('.csv', '')
    baseline_csv = f'baseline_comparison_{timestamp}.csv'
    baselines = read_baseline_csv(baseline_csv)
    
    print(f"✓ Đã đọc {len(rewards)} episodes từ {rewards_csv}")
    if baselines:
        print(f"✓ Đã đọc {len(baselines)} baselines từ {baseline_csv}")
    
    # Vẽ đồ thị
    output_filename = f'plot_{timestamp}.png'
    plot_results(rewards, baselines, output_filename)

