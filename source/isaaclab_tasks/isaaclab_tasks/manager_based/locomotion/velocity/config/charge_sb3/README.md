# Charge Robot 專案 - Sim-to-Real 自主導航系統

## 📋 專案概述

這是一個基於 **深度強化學習（DRL）** 的自主導航機器人專案，使用 **端到端學習** 訓練 Charge 機器人在複雜環境中進行無地圖導航。

### 核心目標

打造一個 **「不需要地圖、能看懂環境、會自己走」** 的智慧導航系統：
- ✅ **無地圖導航**：不依賴 SLAM 建圖，直接從感知到控制
- ✅ **端到端學習**：LiDAR + State → 神經網路 → 速度指令
- ✅ **Sim-to-Real**：在 NVIDIA Isaac Sim 中訓練，部署到真實機器人
- ✅ **快速反應**：反應時間 < 20ms，具備人類般的直覺反應

---

## 🏗️ 系統架構

### 端到端架構

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
│   感知層        │    │   決策層        │    │   執行層      │
│  (Perception)   │───▶│  (Policy Net)   │───▶│  (Robot)     │
│                 │    │                 │    │              │
│  • 2D LiDAR     │    │  • MLP Network  │    │  • 差速驅動   │
│  • Robot State  │    │  • PPO Algorithm│    │  • Charge    │
│  • Goal Info    │    │                 │    │    Robot     │
└─────────────────┘    └─────────────────┘    └──────────────┘
     140 維觀測             [v, ω]             實際運動
```

### 層級式架構（AIT* + RL）

```
┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐
│   AIT* 全域     │    │   Local Goal    │    │    RL PPO    │
│   規劃器        │───▶│   提取器        │───▶│    控制器    │
│   (大腦)        │    │   (介面層)      │    │   (小腦)     │
│   1 Hz          │    │   10 Hz         │    │   50 Hz      │
└─────────────────┘    └─────────────────┘    └──────────────┘
```

---

## 🎯 機器人規格

### Charge 機器人

| 項目 | 規格 |
|------|------|
| **類型** | 差速驅動（Differential Drive） |
| **傳感器** | 2D LiDAR (360°, 131 射線) |
| **執行器** | 2 個主驅動輪 + 2 個萬向輔助輪 |
| **最大速度** | 線速度 1.0 m/s，角速度 2.0 rad/s |
| **控制頻率** | 50 Hz |

### 觀測空間（131 維，Phase 3）

```python
observation = [
    lidar_scan,           # [72]  LiDAR 掃描（360°/5°=72 條射線）
    speed,                # [2]   底盤線速度 (vx, vy)
    goal_position,        # [2]   目標相對位置
    goal_distance,        # [1]   到目標距離
    time_remaining,       # [1]   剩餘時間
    alive_flag,           # [1]   存活標誌
    actions,              # [2]   上一步動作
    obstacles,            # [50]  障礙物資訊（10×5，Phase 3 為 padding）
]
```

### 動作空間

```python
action = [linear_velocity, angular_velocity]
# linear_velocity:  [-1.0, 1.0] m/s
# angular_velocity: [-2.0, 2.0] rad/s
```

---

## 🎓 課程學習策略

### 訓練階段

| Phase | 環境 | 障礙物 | 目標距離 | 學習目標 | 狀態 |
|:-----:|------|:------:|:--------:|---------|:----:|
| **0** | 基礎運動 | 無 | 3-8m | 控制輪子 | ✅ |
| **1** | 靜態避障 | 3 個 | 3-8m | 繞過障礙 | ✅ |
| **2** | 複雜靜態 | 5 個 | 3-8m | 複雜避障 | ⚠️ |
| **3** | 長程導航 | 無 | 2-15m | 導航意志 | 🔄 |
| **4** | 動態環境 | 動態 | 混合 | 最終型態 | - |

### Phase 3：自適應課程學習

```
成功率 > 80% ───▶ 升級（距離 +1m）
     │
成功率 50-80% ───▶ 維持
     │
成功率 < 50% ───▶ 降級（距離 -1m）
```

---

## 🧠 強化學習設計

### 演算法：PPO

- **樣本效率高**
- **穩定性好**
- **易於調參**

### 獎勵函數

```python
# Phase 3 獎勵配置
class RewardsCfg:
    reaching_goal = RewTerm(func=reaching_goal, weight=500.0)
    velocity_to_goal = RewTerm(func=velocity_toward_goal, weight=3.0)
    distance_progress = RewTerm(func=progress_to_goal, weight=5.0)
    tipped_over = RewTerm(func=tipped_over_penalty, weight=-200.0)
    time_out = RewTerm(func=time_out_penalty, weight=-0.05)
```

### 終止條件

```python
# 成功：距離 < 0.3m
# 失敗：翻車、碰撞、超時 (500 步)
```

---

## 🚀 訓練指南

### 環境 ID

```python
# Phase 0
"Isaac-Navigation-Charge-Phase0"

# Phase 1-3 (SB3)
"Isaac-Navigation-Charge-SB3-v0"  # 無障礙
"Isaac-Navigation-Charge-SB3-v1"  # 3 個障礙
"Isaac-Navigation-Charge-SB3-v2"  # 5 個障礙
"Isaac-Navigation-Charge-SB3-v3"  # 長程導航

# 層級式
"Isaac-Navigation-Charge-Hierarchical-v0"
```

### 訓練命令

```bash
# Phase 3 訓練
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v3 \
    --num_envs 256 --headless

# 測試模型
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Navigation-Charge-SB3-Play-v3 \
    --load_path path/to/model.zip
```

---

## 📁 專案結構

```
charge_sb3/
├── __init__.py              # 環境註冊
├── cfg/                     # 環境配置
│   ├── charge_cfg.py        # 機器人配置
│   ├── charge_env_cfg_*.py  # Phase 0-3 配置
│   └── charge_env.py        # Hierarchical 環境
├── agents/                  # PPO 配置
│   └── sb3_ppo_cfg*.yaml
├── mdp/                     # MDP 模組
│   ├── actions/             # 動作空間
│   ├── observations/        # 觀測空間
│   ├── rewards/             # 獎勵函數
│   ├── terminations/        # 終止條件
│   ├── events/              # 課程學習
│   └── path_planner/        # AIT* 規劃器
├── hierarchical/            # 層級式導航
├── md/                      # 項目文檔
└── README.md                # 本文件
```

---

## 🔧 開發指南

### 修改觀測

```python
# 1. 在 mdp/observations/functions.py 定義
# 2. 在 mdp/observations/__init__.py 導出
# 3. 在 cfg/charge_env_cfg_*.py 使用
```

### 修改獎勵

```python
# 1. 在 mdp/rewards/your_file.py 定義
# 2. 在 mdp/rewards/__init__.py 導出
# 3. 在 cfg/charge_env_cfg_*.py 使用
```

---

## 🐛 常見問題

| 問題 | 解決方案 |
|------|----------|
| 蛇行行為 | 提高目標權重，降低避障懲罰 |
| 翻車騙分 | 使用姿態門控 |
| 恐懼障礙 | Phase 3 先移除障礙 |
| PPO std=0 | 檢查觀測歸一化 |

---

## 📊 性能基準

| 指標 | Phase 0 | Phase 1 | Phase 2 | Phase 3 |
|------|:-------:|:-------:|:-------:|:-------:|
| 成功率 | >90% | >70% | >50% | >80% |
| Episode 長度 | <200 | <300 | <400 | <500 |

---

## 📚 參考資料

- [IsaacLab](https://isaac-sim.github.io/IsaacLab/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [PPO 論文](https://arxiv.org/abs/1707.06347)

更多文檔見 `md/` 目錄。

---

**最後更新**: 2026-02-05
