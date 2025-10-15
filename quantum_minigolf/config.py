from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PerformanceFlags:
    blitting: bool = True
    display_downsample: bool = True
    gpu_viz: bool = False
    low_dpi: bool = True
    inplace_step: bool = True
    adaptive_draw: bool = True
    path_decimation: bool = True
    event_debounce: bool = True
    fast_blur: bool = True
    pixel_upscale: bool = True

@dataclass
class GameConfig:
    # Grid & time
    Nx: int = 288
    Ny: int = 144
    dx: float = 1.0
    dy: float = 1.0
    dt: float = 0.35
    steps_per_shot: int = 240
    max_steps_per_shot: int = 2048
    draw_every: int = 3

    # Potentials
    V_edge: float = 200.0
    V_wall: float = 80.0

    # Hole/ball
    hole_r: int = 9
    ball_start_x_frac: float = 0.25

    # Geometry (center barriers)
    single_wall_width: int = 3
    single_wall_thickness_factor: float = 1.0
    slit_height: int = 12
    slit_sep: int = 28
    center_wall_width: int = 6

    # Absorbing boundary
    absorb_width: int = 16
    absorb_strength: float = 0.04

    # Measurement
    quantum_measure: bool = True
    measure_gamma: float = 1.0
    sink_rule: str = "prob_threshold"  # "prob_threshold" | "measurement"

    # Controls
    enable_mouse_swing: bool = True  # True enables mouse hits, False is standard for VR

    # Tunneling / energy mapping
    kmin_frac: float = 0.15     # min fraction of k_max used for very slow swings
    kmax_frac: float = 0.90     # max fraction of k_max to avoid aliasing
    tunneling_speed_weight: float = 0.5   # 0: no speed influence, 1: full influence

    # How strongly barrier height scales with thickness (w/base)^power
    barrier_thickness_power: float = 1.25
    tunneling_thickness_weight: float = 0.9  # 0: no thickness influence, 1: full influence

    # Boundary behaviour
    edge_boundary: str = "reflect"  # "reflect" | "absorb"
    edge_reflect_cells: int = 4
    edge_reflect_height: float = 2000.0

    # Wave initial state
    wave_initial_profile: str = "front"  # "packet" | "front"
    wavefront_transition_len: float = 6.0
    wavefront_sigma_y: float = 6.0
    wavefront_sigma_forward: float = 6.0

    # Shot termination
    shot_stop_mode: str = "time"  # "time" | "friction"
    shot_friction_linear: float = 0.337
    shot_friction_quadratic: float = 0.25
    shot_friction_cubic: float = 0.413
    shot_friction_min_scale: float = 0.01

    sink_prob_threshold: float = 0.25
    measurement_sink_min_prob: float = 1e-3

    # Bias (toward hole)
    boost_hole_probability: bool = True
    boost_hole_probability_factor: float = 0.0
    boost_hole_probability_increment: float = 0.03
    boost_hole_probability_autoincrement_on_measure: bool = True

    # Visibility
    PlotBall: bool = True
    PlotWavePackage: bool = True

    # Visuals/Perf tuning
    smooth_passes: int = 1
    vis_interpolation: str = "bilinear"
    display_downsample_factor: int = 2
    low_dpi_value: int = 72
    target_fps: float = 30
    debounce_ms: int = 12
    path_decimation_stride: int = 3
    overlay_every: int = 3

    # Motion scaling
    movement_speed_scale: float = 2.0  # Multiplier applied to ball/wave travel after a shot

    # Tracker integration
    use_tracker: bool = True
    tracker_speed_scale: float = 0.012
    tracker_threshold: int = 60
    tracker_length_scale: float = 1.0
    tracker_thickness_scale: float = 1.0
    tracker_min_span_px: float = 10.0
    tracker_overlay_thickness_px: float = 4.0
    tracker_debug_window: bool = True
    tracker_crop_x1: Optional[int] = None  # Optional camera ROI (pixels); None keeps full width
    tracker_crop_x2: Optional[int] = None
    tracker_crop_y1: Optional[int] = None
    tracker_crop_y2: Optional[int] = None
    tracker_calibration_path: Optional[str] = None
    show_control_panel: bool = True

    # Time limit per shot (simulation seconds); None = no limit
    shot_time_limit: float = 120

    # Video playback configuration
    video_playback_speed: float = 1.5

    # Map + runtime
    map_kind: str = "double_slit"  # default starting map
    res_scale: float = 1.0

    # Swing & visuals constants (kept configurable)
    ball_r: float = 5.0
    indicator_r: float = 3.0
    impact_min_speed: float = 20.0
    swing_power_scale: float = 0.05
    sigma0: float = 5.0
    perf_sigma0: float = 3.5
    perf_steps_factor: float = 0.6

    # Performance flags
    flags: PerformanceFlags = field(default_factory=PerformanceFlags)
