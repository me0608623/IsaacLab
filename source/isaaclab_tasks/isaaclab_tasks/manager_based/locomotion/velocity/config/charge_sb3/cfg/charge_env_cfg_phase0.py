# ============================================================================
# Phase 0: 車輛動力學與控制校準 (Vehicle Dynamics & Control Calibration)
# ============================================================================
"""
Phase 0 - 車輛動力學校準環境

這是第一個訓練階段，專注於「學會開車」而非「學會導航」。

核心目標：
1. 車輛動力學校準：理解輸入 (油門 v, 方向盤 ω) 與輸出 (速度) 的關係
2. 旋轉 vs 前進：學會何時該原地旋轉、何時該前進
3. 煞車距離：學會在接近目標時減速
4. 動作平滑：避免輪子左右狂抖 (Jerk)

環境特點：
- 16x16m 房間，四面有牆
- 無內部障礙物（專注於運動學）
- 機器人重生在中心 10x10m（確保離牆 3m 以上）
- 目標距離 3-8m（短程直線/大角度轉彎）
- AIT* 規劃直線路徑（空地必須跑 AIT* 測試座標轉換）

獎勵設計（Phase 0 專用）：
- R_progress (weight=20.0): d_{t-1} - d_t，靠近目標就給分
- R_goal (weight=10.0): 抵達目標 d_t < 0.3m
- P_collision (weight=-10.0): 撞牆
- P_smooth (weight=-0.5): ||a_t - a_{t-1}|| 防止抖動
- P_time (weight=-0.05): 每步懲罰，逼它走快點

預期訓練結果：
- >95% 成功率抵達目標
- 無震盪（輪子轉向平滑）
- 路徑效率 > 0.9（實際路徑/直線距離）

訓練命令：
    ./isaaclab.sh -p scripts/reinforcement_learning/sb3/train_charge.py \\
        --task Isaac-Navigation-Charge-Phase0 \\
        --num_envs 256 \\
        --headless \\
        --agent sb3_cfg_entry_point
"""

import math

from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import MultiMeshRayCasterCfg, ContactSensorCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .charge_env_cfg import (
    ChargeNavigationEnvCfg,
    ChargeNavigationEnvCfg_PLAY,
    CommandsCfg,
    EventCfg,
    GoalCommandCfg,
    GOAL_REACH_THRESHOLD,
    ROBOT_BODY_RADIUS,
    TerminationsCfg,
    MySceneCfg,
)

# 導入基礎觀測和獎勵函數
from ..mdp.observations import (
    base_velocity_xy,
    base_angular_velocity_z,
    goal_position_in_robot_frame,
    goal_distance,
    time_remaining_ratio,
    alive_flag,
    lidar_scan_2d_sweep,  # Phase 0 需要 LiDAR
    safe_last_action,  # Phase 0 需要上一步動作
)

# 導入層級式導航觀測（用於 AIT* 局部目標）
from ..mdp.observations.hierarchical_navigation import (
    local_goal_cartesian,  # 局部目標笛卡爾座標（機器人座標系）
)

from ..mdp.rewards import (
    progress_to_goal,
    reaching_goal,
    velocity_toward_goal,
    forward_velocity_reward,
    time_out_penalty,
    lidar_clearance_reward,  # Phase 0 需要 LiDAR 懲罰
    action_rate_penalty,  # Phase 0 核心：平滑度懲罰 (Jerk)
)

from ..mdp.terminations import (
    goal_reached,
    robot_tipped_over,
    wall_collision,  # Phase 0 需要牆壁碰撞檢測
)

# 導入 AIT* 整合事件 ⭐
from ..mdp.events import plan_aitstar_and_update_local_goal
from ..mdp.events import reset_root_state_fixed_per_env


# ============================================================================
# Phase 0 觀測配置（統一 131 維，支持權重遷移）
# ============================================================================

# 導入標準觀測函數（與其他 Phase 一致）
from ..mdp.observations import (
    dynamic_obstacles_state,  # 標準障礙物觀測（10×5=50 維）
)
from .charge_env_cfg import MAX_OBSTACLES  # 10 個障礙物（固定維度）


@configclass
class ObservationsCfgPhase0:
    """Phase 0 觀測配置 - 統一 131 維觀測空間

    ═══════════════════════════════════════════════════════════════════════════
                統一觀測系統 (Unified 131-Dim Observation System)
    ═══════════════════════════════════════════════════════════════════════════

    核心思想：所有 Phase 統一 131 維，確保權重可遷移。

    ┌─────────────────────────────────────────────────────────────────────┐
    │                    統一觀測空間: 131 維                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │ lidar_scan            72 │ 360°/5° = 72 條射線                     │
    │ speed                  2 │ (vx, vy) 線速度                         │
    │ goal_position          2 │ (x, y) 目標相對位置                     │
    │ goal_distance          1 │ 到目標歐式距離                          │
    │ time_remaining         1 │ 剩餘時間比例                            │
    │ alive                  1 │ 存活標誌                                │
    │ actions                2 │ 上一步動作                              │
    │ obstacles            50  │ 10×5 障礙物（padding）                  │
    ├─────────────────────────────────────────────────────────────────────┤
    │ 總計: 72 + 2 + 2 + 1 + 1 + 1 + 2 + 50 = 131 維 (所有 Phase 一致)    │
    └─────────────────────────────────────────────────────────────────────┘

    Phase 0 特點：
    ────────────────────────────────────────────────────────────────────────
    - lidar_scan: 72 維真實數據（掃描四面牆壁）
    - obstacles[50]: 全為 0（無障礙物，用 padding 填充）
    - 其他觀測: 真實數值
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略觀測組 - 131 維統一觀測空間

        ┌─────────────────────────────────────────────────────────────────────┐
        │ 統一觀測空間: 131 維                                                │
        ├─────────────────────────────────────────────────────────────────────┤
        │ lidar_scan            72 │ 360°/5° = 72 條射線                     │
        │ speed                  2 │ (vx, vy) 線速度                         │
        │ goal_position          2 │ (x, y) 目標相對位置                     │
        │ goal_distance          1 │ 到目標歐式距離                          │
        │ time_remaining         1 │ 剩餘時間比例                            │
        │ alive                  1 │ 存活標誌                                │
        │ actions                2 │ 上一步動作                              │
        │ obstacles            50  │ 10×5 障礙物（padding）                  │
        ├─────────────────────────────────────────────────────────────────────┤
        │ 總計: 131 維 (固定)                                                │
        └─────────────────────────────────────────────────────────────────────┘
        """

        # ========================================================================
        # LiDAR 掃描 - 72 維
        # ========================================================================
        lidar_scan = ObsTerm(
            func=lidar_scan_2d_sweep,
            params={"sensor_cfg": SceneEntityCfg("lidar")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # ========================================================================
        # 速度資訊 - 2 維
        # ========================================================================
        speed = ObsTerm(
            func=base_velocity_xy,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        # ========================================================================
        # 目標相對位置 - 2 維
        # ========================================================================
        goal_position = ObsTerm(
            func=goal_position_in_robot_frame,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ========================================================================
        # 目標距離 - 1 維
        # ========================================================================
        goal_distance = ObsTerm(
            func=goal_distance,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # ========================================================================
        # 時間剩餘比例 - 1 維
        # ========================================================================
        time_remaining = ObsTerm(
            func=time_remaining_ratio,
        )

        # ========================================================================
        # 存活標誌 - 1 維
        # ========================================================================
        alive = ObsTerm(
            func=alive_flag,
        )

        # ========================================================================
        # 上一步動作 - 2 維
        # ========================================================================
        actions = ObsTerm(func=safe_last_action)

        # ========================================================================
        # 障礙物資訊 - 50 維（10×5，Phase 0 全為 padding）
        # ========================================================================
        obstacles = ObsTerm(
            func=dynamic_obstacles_state,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "num_obstacles": 0,  # Phase 0: 無障礙物，全為 padding
                "max_obstacles": MAX_OBSTACLES,  # 固定 10 個（產生 50 維）
                "max_distance": 15.0,
            },
        )

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ============================================================================
# Phase 0 獎勵配置（專注於運動學校準）
# ============================================================================

@configclass
class RewardsCfgPhase0:
    """Phase 0 獎勵配置 - 車輛動力學校準

    獎勵設計邏輯（Phase 0 專用）：
    1. R_progress (weight=20.0): d_{t-1} - d_t，靠近目標就給分
    2. R_goal (weight=10.0): 抵達目標 d_t < 0.3m
    3. P_collision (weight=-10.0): 撞牆
    4. P_smooth (weight=-0.5): ||a_t - a_{t-1}|| 防止抖動
    5. P_time (weight=-0.05): 每步懲罰，逼它走快點

    關鍵設計：
    - Progress 權重極高 (20.0)，這是驅動它移動的主力
    - Smooth penalty 防止輪子左右狂抖 (Jerk)
    - Time penalty 強迫它不要原地磨蹭
    """

    # ========== 主要獎勞 ==========

    # R_progress: 進度獎勵（d_{t-1} - d_t）
    # 這是最重要的獎勵！只要靠近目標就給分
    progress_to_goal = RewTerm(
        func=progress_to_goal,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=20.0,  # Phase 0 專用：高權重驅動移動
    )

    # R_goal: 抵達目標獎勵
    reaching_goal = RewTerm(
        func=reaching_goal,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
        },
        weight=10.0,
    )

    # 輔助獎勵：朝向目標的速度分量
    velocity_toward_goal = RewTerm(
        func=velocity_toward_goal,
        params={"asset_cfg": SceneEntityCfg("robot")},
        weight=1.0,
    )

    # ========== 懲罰 ==========

    # P_smooth: 動作平滑懲罰（Jerk Penalty）
    # 懲罰動作變化 ||a_t - a_{t-1}||，防止輪子左右狂抖
    action_smoothness = RewTerm(
        func=action_rate_penalty,
        params={},  # 使用預設 angular_weight=2.0
        weight=-0.5,  # Phase 0 核心：鼓勵平滑控制
    )

    # P_time: 時間懲罰（每步懲罰）
    # 強迫它盡快到達，不要原地磨蹭
    time_out_penalty = RewTerm(
        func=time_out_penalty,
        weight=-0.05,  # 從 -0.01 提高到 -0.05
    )

    # P_collision: 牆壁接近懲罰（基於 LiDAR）
    # 讓神經網路學會使用 LiDAR 數據檢測邊界
    wall_proximity_penalty = RewTerm(
        func=lidar_clearance_reward,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
        },
        weight=0.5,  # 正權重，距離越遠獎勵越高
    )


# ============================================================================
# Phase 0 終止條件配置
# ============================================================================

@configclass
class TerminationsCfgPhase0:
    """Phase 0 終止條件配置

    - goal_reached: 抵達目標 (d < 0.3m)
    - robot_tipped_over: 機器人翻倒
    - wall_collision: 撞牆懲罰（雖然很難撞到，但要有底線）
    """

    # 抵達目標
    goal_reached = DoneTerm(
        func=goal_reached,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": GOAL_REACH_THRESHOLD,
        },
    )

    # 機器人翻倒
    robot_tipped_over = DoneTerm(
        func=robot_tipped_over,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # 牆壁碰撞（Phase 0：雖然很難撞到，但要有底線）
    wall_collision = DoneTerm(
        func=wall_collision,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


# ============================================================================
# Phase 0 場景配置（無障礙物）
# ============================================================================

@configclass
class MySceneCfgPhase0(MySceneCfg):
    """Phase 0 場景配置 - 16x16m 房間，四面牆壁，無內部障礙物

    設計理念：
    - 牆壁提供 LiDAR 邊界值（靠近牆時距離變小）
    - 無內部障礙物，專注於車輛動力學校準
    - AIT* 在空地會規劃直線路徑
    """

    def __post_init__(self):
        """初始化場景（四面牆壁版本）"""
        InteractiveSceneCfg.__post_init__(self)

        # 房間配置
        room_size = 8.0  # 半徑 8m = 16x16m 房間
        wall_thickness = 0.2
        wall_height = 1.5
        wall_length = room_size * 2 + wall_thickness
        wall_color = (0.5, 0.5, 0.5)

        wall_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=True,
            disable_gravity=True,
        )
        wall_collision_props = sim_utils.CollisionPropertiesCfg()
        wall_visual = sim_utils.PreviewSurfaceCfg(
            diffuse_color=wall_color,
            metallic=0.1,
        )

        # 北牆 (Y = +8m)
        self.wall_north = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_North",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, room_size, wall_height / 2),
            ),
        )

        # 南牆 (Y = -8m)
        self.wall_south = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_South",
            spawn=sim_utils.CuboidCfg(
                size=(wall_length, wall_thickness, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, -room_size, wall_height / 2),
            ),
        )

        # 東牆 (X = +8m)
        self.wall_east = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_East",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(room_size, 0.0, wall_height / 2),
            ),
        )

        # 西牆 (X = -8m)
        self.wall_west = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Wall_West",
            spawn=sim_utils.CuboidCfg(
                size=(wall_thickness, wall_length, wall_height),
                rigid_props=wall_rigid_props,
                collision_props=wall_collision_props,
                visual_material=wall_visual,
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(-room_size, 0.0, wall_height / 2),
            ),
        )

        # 覆蓋 lidar 配置（只檢測牆壁，不檢測障礙物）
        # Phase 0 沒有障礙物，所以移除 Obstacle_.* 的 raycast target
        self.lidar = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/charger_rover_urdf5/base_link",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
            ray_alignment="yaw",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=1,
                vertical_fov_range=(0.0, 0.0),
                horizontal_fov_range=(-180.0, 180.0),
                horizontal_res=5.0,  # 72 條射線
            ),
            max_distance=10.0,
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="/World/ground",
                    track_mesh_transforms=False,
                ),
                # Phase 0：只檢測牆壁（不檢測 Obstacle_ 因為沒有內部障礙物）
                MultiMeshRayCasterCfg.RaycastTargetCfg(
                    prim_expr="{ENV_REGEX_NS}/Wall_.*",  # 匹配四面牆
                    track_mesh_transforms=False,
                ),
            ],
            update_period=0.04,
            debug_vis=True,  # 開啟可視化方便調試
        )


# ============================================================================
# Phase 0 命令配置
# ============================================================================

@configclass
class CommandsCfgPhase0(CommandsCfg):
    """Phase 0 命令配置 - 車輛動力學校準

    設計理念：
    - 目標距離 3-8m：強迫練習「直線加速」和「大角度轉彎」
    - 全方向隨機：確保學會 360 度轉向
    """

    # 目標位置命令
    goal_command = GoalCommandCfg(
        asset_name="robot",
        debug_vis=True,  # 顯示紅色箭頭
        resampling_time_range=(5.0, 10.0),  # 每 5-10 秒重新生成目標
        # Phase 0 規格：目標距離 3m ~ 8m
        # 規格書要求：強迫機器人練習「直線加速」和「大角度轉彎」
        ranges=GoalCommandCfg.Ranges(
            distance=(3.0, 8.0),  # 3-8 米（從 2-6m 調整）
            angle=(-math.pi, math.pi),  # 全方向 360 度
        ),
    )


# ============================================================================
# Phase 0 事件配置
# ============================================================================

@configclass
class EventCfgPhase0(EventCfg):
    """Phase 0 事件配置 - 車輛動力學校準

    設計理念：
    - 機器人重生在中心 10x10m 區域（-5 ~ +5）
    - 確保離牆壁至少有 3m 安全距離（16x16m 房間，牆在 ±8m）
    - 隨機朝向 0-360 度
    - AIT* 規劃直線路徑（空地測試座標轉換）
    - Carrot-on-stick: 前方 2m 處的局部目標
    """

    # 重置機器人位置（Phase 0 專用：中心 10x10m 區域）
    reset_base = EventTerm(
        func=reset_root_state_fixed_per_env,
        mode="reset",
        params={
            "pose_range": {
                # Phase 0 規格：機器人重生在中心 10x10m 區域
                # 確保離牆壁至少有 3m 安全距離
                "x": (-5.0, 5.0),  # X 座標範圍：-5 到 5 米（10m 寬）
                "y": (-5.0, 5.0),  # Y 座標範圍：-5 到 5 米（10m 高）
                "yaw": (-3.14, 3.14)  # 隨機朝向 0-360 度
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
            },
        },
    )

    # ========== AIT* 路徑規劃整合 ⭐ ==========
    # 在環境重置時使用 AIT* 規劃路徑並更新局部目標
    # 這讓 RL 學會追隨 AIT* 規劃的「紅蘿蔔」而非直接衝向終點
    plan_aitstar_path = EventTerm(
        func=plan_aitstar_and_update_local_goal,
        mode="reset",
        params={
            "lookahead_distance": 2.0,  # Carrot-on-stick 前瞻距離
            "map_size": (16.0, 16.0),    # 地圖大小
            "robot_radius": 0.3,         # 機器人半徑
            "visualize_path": True,       # 啟用 AIT* 路徑可視化（綠色線條和點）
        },
    )


# ============================================================================
# Phase 0 環境配置主類
# ============================================================================

@configclass
class ChargeNavigationEnvCfgPhase0(ChargeNavigationEnvCfg):
    """Phase 0 導航環境配置 - 車輛動力學校準

    這是第一個訓練階段，專注於「學會開車」：
    1. 車輛動力學：輸入 (v, ω) → 輸出 (速度)
    2. 旋轉 vs 前進：何時該原地旋轉、何時該前進
    3. 煞車距離：接近目標時減速
    4. 動作平滑：避免輪子左右狂抖

    環境規格：
    - 地圖：16x16m 房間，四面牆壁
    - 機器人重生：中心 10x10m（確保離牆 3m+）
    - 目標距離：3-8m
    - 無障礙物

    畢業標準：
    - 成功率 > 95%
    - 無震盪（輪子轉向平滑）
    - 路徑效率 > 0.9

    訓練建議：
    - num_envs: 256-512
    - 訓練步數: 2M-5M steps
    """

    # 場景配置（四面牆，無內部障礙物）
    # env_spacing 需要大於房間尺寸 (16m)，避免場景重疊
    scene: MySceneCfgPhase0 = MySceneCfgPhase0(num_envs=256, env_spacing=18.0)

    # 觀測配置（使用完整觀測，包含 LiDAR）
    observations: ObservationsCfgPhase0 = ObservationsCfgPhase0()

    # 獎勵配置
    rewards: RewardsCfgPhase0 = RewardsCfgPhase0()

    # 終止條件
    terminations: TerminationsCfgPhase0 = TerminationsCfgPhase0()

    # 命令配置
    commands: CommandsCfgPhase0 = CommandsCfgPhase0()

    # 事件配置
    events: EventCfgPhase0 = EventCfgPhase0()

    # Episode 長度
    episode_length_s = 20.0  # 20 秒足夠完成房間內導航


@configclass
class ChargeNavigationEnvCfgPhase0_PLAY(ChargeNavigationEnvCfgPhase0):
    """Phase 0 演示環境配置（單環境，無頭模式關閉）"""

    scene: MySceneCfgPhase0 = MySceneCfgPhase0(num_envs=1, env_spacing=18.0)
    observations: ObservationsCfgPhase0 = ObservationsCfgPhase0()


# ============================================================================
# 導出
# ============================================================================

__all__ = [
    "ChargeNavigationEnvCfgPhase0",
    "ChargeNavigationEnvCfgPhase0_PLAY",
]
