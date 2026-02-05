# Charge Robot 專案 - AI 輔助開發使用說明書

## 專案路徑
```
/home/aa/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/charge_sb3
```

---

## 一、專案概述 (Project Overview)

### 1.1 專案目標
這是一個 **Sim-to-Real（模擬到現實）** 的自主導航機器人專案，使用深度強化學習（DRL）訓練一個名為 **Charge** 的差速驅動機器人，實現：

- **無地圖導航**：不依賴傳統 SLAM 建圖，直接從 LiDAR 感知輸出到控制指令
- **端到端學習**：輸入（LiDAR + 狀態）→ 神經網路 → 輸出（速度指令）
- **動態環境適應**：應對移動障礙物和複雜場景

### 1.2 技術棧
| 組件 | 技術 |
|------|------|
| 模擬器 | NVIDIA Isaac Sim / Isaac Lab |
| 演算法 | Stable-Baselines3 (SB3) PPO / RSL-RL PPO |
| 機器人 | Charge（差速驅動，2D LiDAR） |
| 路徑規劃 | AIT* (層級式導航可選) |

---

## 二、專案結構 (Project Structure)

### 2.1 目錄結構
```
charge_sb3/
├── __init__.py              # Gym 環境註冊（重要！定義所有可用的環境 ID）
├── cfg/                     # 環境配置
│   ├── charge_cfg.py                    # 機器人物理配置
│   ├── charge_env_cfg.py                # Phase 1 環境配置
│   ├── charge_env_cfg_v2.py             # Phase 2 環境配置
│   ├── charge_env_cfg_v3.py             # Phase 3 環境配置（自適應課程）
│   ├── charge_env_cfg_phase0.py         # Phase 0 配置（基礎運動學）
│   ├── charge_env.py                    # 環境類（HierarchicalChargeNavigationEnv）
│   └── __init__.py
│
├── agents/                  # 演算法配置
│   ├── sb3_ppo_cfg.yaml                  # SB3 PPO 基礎配置
│   ├── sb3_ppo_cfg_v2.yaml               # SB3 PPO Phase 2 配置
│   ├── sb3_ppo_cfg_v3.yaml               # SB3 PPO Phase 3 配置
│   ├── rsl_rl_ppo_cfg.py                 # RSL-RL PPO 基礎配置
│   └── __init__.py
│
├── mdp/                     # MDP（Markov Decision Process）模組
│   ├── __init__.py                       # 統一導出所有 MDP 函數
│   ├── actions/                          # 動作空間
│   │   └── differential_drive.py         # 差速驅動動作
│   ├── observations/                     # 觀測空間
│   │   ├── functions.py                  # 基礎觀測函數
│   │   ├── utils.py                      # 觀測工具
│   │   └── fixed_topology.py             # 固定拓撲觀測系統（122維）
│   ├── rewards/                          # 獎勵函數
│   │   ├── goal_rewards.py               # 目標相關獎勵
│   │   ├── safety_rewards.py             # 安全相關獎勵
│   │   ├── motion_rewards.py             # 運動相關獎勵
│   │   └── utils.py
│   ├── terminations/                     # 終止條件
│   │   ├── collision.py
│   │   ├── goal.py
│   │   └── robot_state.py
│   ├── events/                           # 事件處理
│   │   ├── curriculum.py                 # 課程學習管理器
│   │   ├── curriculum_strategies.py      # 課程學習策略
│   │   ├── goal_distance_curriculum.py   # 目標距離課程
│   │   ├── reset.py
│   │   └── obstacles.py
│   ├── path_planner/                     # 路徑規劃器（AIT*）
│   │   ├── aitstar_planner.py
│   │   ├── frenet_transform.py           # Frenet 座標轉換
│   │   ├── local_goal_extractor.py       # 局部目標提取
│   │   └── environment_map.py
│   └── core/                             # 核心狀態管理
│       ├── state.py
│       └── types.py
│
├── curriculum/              # 課程學習配置
├── domain_randomization/    # 域隨機化（實驗中）
├── globle_planner/          # 全局規劃器相關
├── hierarchical/            # 層級式導航相關
├── wrappers/                # 環境包裝器
├── docs/                    # 設計文檔
├── md/                      # 項目說明文檔
├── paper/                   # 參考論文
├── test/                    # 測試文件
├── scripts/                 # 腳本工具
└── readme/                  # 開發日誌
```

---

## 三、訓練階段 (Curriculum Phases)

### 3.1 課程學習策略

專案採用 **由簡入繁** 的課程學習策略：

| 階段 | 環境描述 | 障礙物 | 目標距離 | 當前狀態 |
|:----:|---------|:------:|:--------:|:--------:|
| **Phase 0** | 基礎運動學 | 無 | 3-8m | 基礎能力訓練 |
| **Phase 1** | 靜態避障 | 3個靜態 | 3-8m | ✅ 完成 |
| **Phase 2** | 複雜靜態 | 5個靜態 | 3-8m | ⚠️ 發現蛇行問題 |
| **Phase 3** | 長程導航 | 無 | 2-15m（自適應）| 🔄 進行中 |
| **Phase 4** | 動態環境 | 動態障礙 | 混合 | 待開始 |

### 3.2 Phase 0 - 基礎運動學
- **目標**：學會控制輪子、理解輸入輸出關係
- **環境**：空曟房間（16x16m），四面牆壁
- **觀測**：最小化（goal_position, velocity, time, actions）
- **無 LiDAR**：專注於運動控制

### 3.3 Phase 3 - 自適應課程學習（當前重點）
- **目標**：建立強大的「導航意志」，消除恐懼
- **特點**：
  - 無障礙物，專注長距離導航
  - **自適應距離**：根據成功率自動調整目標距離（2m-15m）
  - 升級條件：成功率 > 80%
  - 降級條件：成功率 < 50%

---

## 四、已註冊的 Gym 環境 (Registered Environments)

所有環境註冊在 `__init__.py` 中：

### 4.1 Phase 0 環境
```python
# 訓練環境
"Isaac-Navigation-Charge-Phase0"

# 測試/演示環境
"Isaac-Navigation-Charge-Phase0-Play"
```

### 4.2 Phase 1-3 環境（RSL-RL）
```python
# v0: 無障礙物（Phase 3 配置）
"Isaac-Navigation-Charge-v0"

# v1: Phase 1（3個靜態障礙物）
"Isaac-Navigation-Charge-v1"

# v2: Phase 2（5個靜態障礙物）
"Isaac-Navigation-Charge-v2"

# v3: Phase 3（目標距離課程，無障礙物）
"Isaac-Navigation-Charge-v3"

# Play 版本（後綴 -Play-）
"Isaac-Navigation-Charge-Play-v0"
"Isaac-Navigation-Charge-Play-v1"
"Isaac-Navigation-Charge-Play-v2"
"Isaac-Navigation-Charge-Play-v3"
```

### 4.3 Stable-Baselines3 環境
```python
# SB3 版本（後綴 -SB3-）
"Isaac-Navigation-Charge-SB3-v0"
"Isaac-Navigation-Charge-SB3-v1"
"Isaac-Navigation-Charge-SB3-v2"
"Isaac-Navigation-Charge-SB3-v3"

# Play 版本
"Isaac-Navigation-Charge-SB3-Play-v0"
"Isaac-Navigation-Charge-SB3-Play-v1"
"Isaac-Navigation-Charge-SB3-Play-v2"
"Isaac-Navigation-Charge-SB3-Play-v3"
```

### 4.4 層級式導航環境（帶 AIT*）
```python
# 使用工廠函數創建
"Isaac-Navigation-Charge-Hierarchical-v0"  # 無障礙物
"Isaac-Navigation-Charge-Hierarchical-v1"  # 3個靜態障礙物
```

---

## 五、關鍵配置說明 (Key Configurations)

### 5.1 機器人配置 (`charge_cfg.py`)
```python
CHARGE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/aa/usd/charge/charge.usd",  # 3D 模型路徑
        activate_contact_sensors=True,               # 啟用碰撞檢測
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "left_wheel_joint"],
            stiffness=0.0,          # 純速度控制
            damping=10.0,           # 適中響應
            velocity_limit=100.0,   # 最大角速度
        ),
    },
)
```

### 5.2 環境配置結構 (`charge_env_cfg_*.py`)

每個環境配置文件都包含：
- **`MySceneCfg`**: 場景配置（機器人、牆壁、障礙物、目標）
- **`ObservationsCfg`**: 觀測空間配置
- **`ActionsCfg`**: 動作空間配置
- **`RewardsCfg`**: 獎勵函數配置
- **`TerminationsCfg`**: 終止條件配置
- **`EventCfg`**: 事件處理配置（課程學習）
- **`CommandsCfg`**: 目標命令配置

### 5.3 SB3 PPO 配置 (`agents/sb3_ppo_cfg*.yaml`)

```yaml
# sb3_ppo_cfg.yaml 示例
algorithm:
  class_name: PPO
  policy: "MlpPolicy"

# 網路架構
policy_kwargs:
  net_arch:
    - [256, 256, 128]  # [pi_layers, vf_layers]
  activation_fn: "relu"

# 訓練參數
learning_rate: 3.0e-4
n_steps: 2048
batch_size: 512
n_epochs: 10
gamma: 0.99
gae_lambda: 0.95

# PPO 特有參數
clip_range: 0.2
ent_coef: 0.0
vf_coef: 0.5
max_grad_norm: 1.0
```

---

## 六、MDP 模組說明 (MDP Modules)

### 6.1 觀測函數 (`mdp/observations/`)

| 函數 | 描述 | 維度 |
|------|------|:----:|
| `goal_position_in_robot_frame` | 機器人座標系中的目標位置 | 2 |
| `goal_distance` | 到目標的歐式距離 | 1 |
| `base_velocity_xy` | 底盤線速度 | 2 |
| `base_angular_velocity_z` | 底盤角速度 | 1 |
| `time_remaining_ratio` | 剩餘時間比例 | 1 |
| `alive_flag` | 存活標誌 | 1 |
| `heading_error_to_goal` | 朝向目標的誤差角 | 1 |
| `lidar_scan` | LiDAR 掃描數據 | 131 |
| `safe_last_action` | 上一步動作（夾斷） | 2 |
| `dynamic_obstacles_state` | 動態障礙物狀態 | 變動 |

### 6.2 獎勵函數 (`mdp/rewards/`)

**目標獎勵** (`goal_rewards.py`)：
- `velocity_toward_goal`: 速度在目標方向的投影
- `progress_to_goal`: 距離減少量
- `reaching_goal`: 抵達目標的獎勵
- `approaching_goal_bonus`: 接近目標的額外獎勵
- `heading_to_goal`: 朝向目標的獎勵

**安全獎勵** (`safety_rewards.py`)：
- `obstacle_avoidance_reward`: 避障獎勵
- `collision_penalty`: 碰撞懲罰
- `wall_collision_penalty`: 牆壁碰撞懲罰
- `safe_navigation_bonus`: 安全導航獎勵

**運動獎勵** (`motion_rewards.py`)：
- `forward_velocity_reward`: 前進速度獎勵
- `forward_motion_reward`: 向前移動獎勵
- `action_rate_penalty`: 動作變化懲罰（防止抖動）

**路徑跟隨獎勵**（層級式導航）：
- `progress_along_path`: 沿路徑進度
- `cross_track_error`: 橫向誤差
- `path_direction_reward`: 路徑方向獎勵

### 6.3 終止條件 (`mdp/terminations/`)

- `goal_reached`: 抵達目標（成功）
- `robot_tipped_over`: 機器人翻車（失敗）
- `robot_flying`: 機器人飛出（失敗）
- `wall_collision`: 牆壁碰撞（失敗）

### 6.4 課程學習管理器 (`mdp/events/curriculum.py`)

```python
class CurriculumManager:
    """自適應課程學習管理器"""

    def update(self, metrics: CurriculumMetrics) -> CurriculumAction:
        """
        根據當前指標決定課程動作：
        - success_rate > 0.8: INCREASE（增加難度）
        - success_rate < 0.5: DECREASE（降低難度）
        - 其他: MAINTAIN（維持）
        """
```

---

## 七、常用開發操作 (Common Operations)

### 7.1 修改觀測空間

1. 在 `mdp/observations/functions.py` 中定義新函數
2. 在 `mdp/observations/__init__.py` 中導出
3. 在 `cfg/charge_env_cfg_*.py` 中的 `ObservationsCfg` 中添加：

```python
class ObservationsCfg:
    policy = ObsGroup(
        observations={
            "your_new_obs": ObsTerm(func=your_new_function),
        },
    )
```

### 7.2 修改獎勵函數

1. 在 `mdp/rewards/` 中定義新函數
2. 在 `mdp/rewards/__init__.py` 中導出
3. 在 `cfg/charge_env_cfg_*.py` 中的 `RewardsCfg` 中添加：

```python
class RewardsCfg:
    your_new_reward = RewTerm(func=your_reward_function, weight=1.0)
```

### 7.3 修改終止條件

類似獎勵函數，在 `TerminationsCfg` 中添加：

```python
class TerminationsCfg:
    your_new_termination = DoneTerm(func=your_termination_function)
```

### 7.4 創建新的環境版本

1. 複製現有配置文件（如 `charge_env_cfg_v3.py`）
2. 修改 `class ChargeNavigationEnvCfgV4`
3. 在 `cfg/__init__.py` 中導出
4. 在主 `__init__.py` 中註冊新環境：

```python
gym.register(
    id="Isaac-Navigation-Charge-SB3-v4",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg.charge_env_cfg_v4:ChargeNavigationEnvCfgV4",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
```

---

## 八、訓練與測試 (Training & Testing)

### 8.1 訓練命令

```bash
# 使用 SB3 訓練
./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \
    --task Isaac-Navigation-Charge-SB3-v0 \
    --num_envs 256 \
    --headless

# 使用 RSL-RL 訓練
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Navigation-Charge-v0 \
    --num_envs 4096
```

### 8.2 測試訓練好的模型

```bash
# 加載模型並測試
./isaaclab.sh -p scripts/reinforcement_learning/sb3/play.py \
    --task Isaac-Navigation-Charge-SB3-Play-v0 \
    --num_envs 1 \
    --load_path path/to/model.zip
```

### 8.3 輸出目錄

訓練輸出保存在：
```
~/isaaclab/isaac-navigator-charge-sb3-v0/
├── logs/           # TensorBoard 日誌
├── model/          # 模型檢查點
└── results/        # 評估結果
```

---

## 九、已知問題與解決方案 (Known Issues)

### 9.1 蛇行問題（Phase 2）
**問題**：Agent 出現過度蛇行，恐懼障礙物
**原因**：碰撞懲罰權重過高，目標獎勵不足以驅動導航
**解決**：
1. 降低碰撞懲罰權重
2. 增加朝向目標的獎勵
3. 使用姿態門控（翻車則取消移動獎勵）

### 9.2 翻車騙獎勵
**問題**：Agent 故意翻車滑行以避免碰撞
**解決**：在 `motion_rewards.py` 中實現姿態門控

### 9.3 PPO std >= 0 警告
**問題**：觀測值包含 NaN 或 Inf
**解決**：使用自定義環境類 `ChargeNavigationEnv` 進行觀測檢查

---

## 十、重要文件位置速查 (Quick Reference)

| 需求 | 文件路徑 |
|------|---------|
| 環境註冊 | `__init__.py` |
| 機器人配置 | `cfg/charge_cfg.py` |
| Phase 0 配置 | `cfg/charge_env_cfg_phase0.py` |
| Phase 1 配置 | `cfg/charge_env_cfg.py` |
| Phase 2 配置 | `cfg/charge_env_cfg_v2.py` |
| Phase 3 配置 | `cfg/charge_env_cfg_v3.py` |
| SB3 PPO 參數 | `agents/sb3_ppo_cfg.yaml` |
| 觀測函數 | `mdp/observations/functions.py` |
| 獎勵函數 | `mdp/rewards/` |
| 終止條件 | `mdp/terminations/` |
| 課程學習 | `mdp/events/curriculum.py` |
| 層級式環境 | `cfg/charge_env.py` (HierarchicalChargeNavigationEnv) |

---

## 十一、開發檢查清單 (Development Checklist)

當接到新任務時，AI 應該：

1. **確認目標 Phase**（0/1/2/3）
2. **閱讀對應配置文件**（`charge_env_cfg_v*.py`）
3. **理解當前獎勵函數**（`RewardsCfg`）
4. **理解當前觀測空間**（`ObservationsCfg`）
5. **確認要修改的模組**（觀測/獎勵/終止/事件）
6. **修改後確保導出**（在 `__init__.py` 中添加）
7. **考慮向後兼容性**

---

## 十二、專案設計理念 (Design Philosophy)

### 核心原則
1. **課程學習**：由簡入繁，循序漸進
2. **模組化設計**：MDP 組件高度解耦
3. **可擴展性**：易於添加新的觀測/獎勵/階段
4. **Auto-Tuning**：自適應課程自動調整難度
5. **安全第一**：碰撞懲罰確保 Agent 學會避障

### 當前策略 (2026-02-05)
- **專注 Phase 3**：建立強大的基礎導航能力
- **自適應課程**：根據成功率自動調整目標距離
- **未來 Phase 4**：結合導航意志與避障能力
