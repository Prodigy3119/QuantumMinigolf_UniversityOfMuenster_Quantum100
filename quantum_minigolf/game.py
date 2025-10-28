from __future__ import annotations
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import os

try:
    from matplotlib.widgets import Slider  # type: ignore
except Exception:
    Slider = None  # type: ignore

from .tracker import TrackerManager, TrackerConfig
from .calibration import CalibrationData
from .config import GameConfig
from .backends import Backend
from .course import Course
from .physics import (
    step_wave, prepare_frame, compute_expectation, covariance_ellipse, sample_from_density
)
from .visuals import Visuals

try:  # Matplotlib Qt helper (available when using QtAgg backend)
    from matplotlib.backends.qt_compat import QtWidgets  # type: ignore
except Exception:  # pragma: no cover - backend specific
    QtWidgets = None  # type: ignore


class QuantumMiniGolfGame:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self._perf_enabled = bool(getattr(self.cfg, 'performance_increase', False))
        self.be = Backend()
        if self.be.USE_GPU:
            self.cfg.flags.gpu_viz = True
        else:
            self.cfg.flags.gpu_viz = False
        self._performance_theta = float(getattr(self.cfg, 'performance_theta', 0.6 * math.pi))
        self._dt_tolerance = float(max(1e-9, getattr(self.cfg, 'performance_dt_tolerance', 1e-6)))
        self._perf_drift_threshold = float(max(0.0, getattr(self.cfg, 'performance_drift_threshold', 1e-5)))
        self._perf_max_drift_steps = int(max(1, getattr(self.cfg, 'performance_max_drift_steps', 4)))
        self._perf_fft_plan_cache: dict[tuple[tuple[int, ...], str], object] = {}
        self._perf_expK_powers: dict[int, object] = {}
        self._perf_drift_fail_logged = False
        self._perf_window_fail_logged = False
        self._perf_dt_fail_logged = False
        self._perf_fast_mode_logged = False
        self._perf_fft_plan_fail_logged = False
        self._perf_friction_expV = None
        self._perf_friction_expK = None
        self._perf_friction_scales = None
        if self.cfg.fast_mode_paraxial and not self._perf_enabled:
            print("[perf] fast-mode-paraxial requires --performance-increase; ignoring.")
            self.cfg.fast_mode_paraxial = False

        self._trim_visuals = bool(self._perf_enabled and getattr(self.cfg, 'auto_trim_visuals', True))
        if self._trim_visuals:
            if getattr(self.cfg.flags, 'render_paths', True):
                self.cfg.flags.render_paths = False
            self.cfg.flags.minimal_annotations = True
            self.cfg.flags.path_decimation = True

        # numeric dtypes & grids
        self.f32 = np.float32
        self.c64 = np.complex64

        # course
        self.course = Course(cfg, self.be)
        self.Nx, self.Ny = self.course.Nx, self.course.Ny
        self._refresh_course_assets(update_visuals=False)

        # k-space (CPU -> XP)
        self.k2, self.k_max = self.be.build_kgrid(
            self.Nx, self.Ny, cfg.dx, cfg.dy)
        self._k2_cpu_max = float(np.max(self.be.to_cpu(self.k2))) if self.k2.size else 0.0

        # grids for expectations
        Xc, Yc = np.meshgrid(np.arange(self.Nx, dtype=np.float32),
                             np.arange(self.Ny, dtype=np.float32), indexing='xy')
        self.Xgrid = self.be.to_xp(Xc, np.float32)
        self.Ygrid = self.be.to_xp(Yc, np.float32)

        # exponents for current dt
        self._current_exp_dt: float | None = None
        self._ensure_exponents(cfg.dt, force=True)
        self._perf_window_enabled = bool(self._perf_enabled and getattr(self.cfg, 'performance_enable_window', False))
        self._perf_window_margin = int(max(4, getattr(self.cfg, 'performance_window_margin', 12)))

        # positions
        start_x = int(
            round(max(8, min(self.Nx - 8, self.Nx * self.cfg.ball_start_x_frac))))
        self.ball_pos = np.array(
            [float(start_x), self.Ny / 2], dtype=np.float32)
        self._multiple_shots_enabled = bool(getattr(self.cfg, 'multiple_shots', False))
        self._shot_count = 0

        # visuals
        overlay_initial = bool(getattr(self.cfg, 'tracker_overlay_initial', False))

        self.viz = Visuals(self.Nx, self.Ny, self.hole_center,
                           self.cfg.hole_r, self.cfg.flags, self.cfg)
        if self._trim_visuals or getattr(self.cfg.flags, 'minimal_annotations', False) or not getattr(self.cfg.flags, 'render_paths', True):
            self.viz.apply_performance_trim(self.cfg.flags)
        if not overlay_initial:
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)
        self.viz.set_course_patches(self.course.course_patches)
        self._refresh_course_assets()
        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])
        self.viz.update_title(self._title_text())
        self._update_shot_counter()

        self._background_cycle = self._build_background_cycle()
        self._background_index = 0
        current_bg = self.viz.current_background_path()
        if current_bg is not None:
            idx = self._find_background_index(current_bg)
            if idx is None:
                self._background_cycle.insert(1, current_bg)
                self._background_index = 1
            else:
                self._background_index = idx
        elif getattr(self.cfg, 'background_image_path', None):
            cfg_path = Path(str(self.cfg.background_image_path)).expanduser()
            idx = self._find_background_index(cfg_path)
            if idx is None:
                try:
                    resolved = cfg_path.resolve()
                except Exception:
                    resolved = cfg_path
                self._background_cycle.insert(1, resolved)
                self._background_index = 1
            else:
                self._background_index = idx
        self.tracker = None
        self.tracker_cfg = None
        self._tracker_timer = None
        self._tracker_decoupled = False
        self._tracker_overlay_initial = overlay_initial
        self._tracker_overlay_enabled = overlay_initial
        self._tracker_force_disabled = False
        self._tracker_area_valid = True
        self._tracker_last_area = 0.0
        self._tracker_auto_scale = bool(getattr(self.cfg, 'tracker_auto_scale', True))
        self._tracker_corr_scale_x = 1.0
        self._tracker_corr_offset_x = 0.0
        self._tracker_corr_scale_y = 1.0
        self._tracker_corr_offset_y = 0.0
        self._tracker_min_x: float | None = None
        self._tracker_max_x: float | None = None
        self._tracker_min_y: float | None = None
        self._tracker_max_y: float | None = None
        self._tracker_base_length_px: float | None = None
        self._tracker_base_thickness_px: float | None = None
        self._tracker_size_scale = float(max(0.1, getattr(self.cfg, 'tracker_length_scale', 0.3)))
        base_speed = float(max(1e-6, getattr(self.cfg, 'tracker_speed_base', getattr(self.cfg, 'tracker_speed_scale', 0.01))))
        self._tracker_speed_base = base_speed
        try:
            self.cfg.tracker_speed_base = base_speed
        except Exception:
            pass
        self._abort_shot_requested = False
        self._debug_log_enabled = bool(
            getattr(self.cfg, 'use_tracker', False)
            and getattr(self.cfg, 'log_data', False)
        )
        self._debug_log_path = Path(getattr(self.cfg, 'tracker_debug_log_path', 'vr_debug_log.txt'))
        self._debug_log_timer = None
        self._debug_prev_putter: dict[str, object] = {"center": None, "time": None}
        self._debug_prev_tracker: dict[str, object] = {"center": None, "time": None}
        self._debug_session_started = False
        self._debug_log_write_failed = False
        self._display_info_timer = None
        tracker_enabled = bool(getattr(self.cfg, 'use_tracker', False))
        if bool(getattr(self.cfg, 'enable_mouse_swing', False)):
            self._set_tracker_force_disabled(True)
        self._init_tracker(tracker_enabled)
        self.viz.fig.canvas.mpl_connect('close_event', self._on_close)

        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        else:
            self.viz.fig.canvas.draw_idle()

        # game state
        self.game_over = False
        self.shot_in_progress = False
        self.show_info = False
        self.show_interference = False
        self._last_density_cpu: np.ndarray | None = None
        self._last_measure_xy: tuple[float, float] | None = None
        self._last_measure_prob: float | None = None
        self._last_ex = None
        self._last_ey = None
        self._last_phase_cpu = None
        self._interference_profile: np.ndarray | None = None
        self._frame_listeners: list[Callable[["QuantumMiniGolfGame"], None]] = []
        self._playback_timer = None
        self._playback_iter = None
        self._playback_image = None
        self._playback_path: Path | None = None
        self._playback_close = None
        self._playback_hold = False
        self._video_playback_speed = float(max(0.1, getattr(self.cfg, 'video_playback_speed', 1.0)))
        self._wavefront_profile = str(getattr(self.cfg, 'wave_initial_profile', 'packet')).lower()
        self._wavefront_active = self._wavefront_profile == 'front'
        self._wavefront_dir = np.array([1.0, 0.0], dtype=np.float32)
        self._wavefront_kmag = 0.0
        self._overlay_compute_during_shot = False
        self._pending_info_overlay = False
        self._pending_measure_after_shot = False

        # swing detection
        self.cursor_inside = False
        self.indicator_pos = None
        self.last_mouse_pos = None
        self.last_mouse_t = None

        self.ds_factor = int(max(1, self.cfg.display_downsample_factor))
        self.path_decim = int(max(1, self.cfg.path_decimation_stride))
        self._render_paths = bool(getattr(self.cfg.flags, 'render_paths', True))
        self._minimal_annotations = bool(getattr(self.cfg.flags, 'minimal_annotations', False))

        # perf-mode values (you can expose a toggle later)
        self.perf_mode = bool(self._perf_enabled)
        perf_draw_base = max(self.cfg.draw_every, int(getattr(self.cfg, 'performance_draw_every', 3)))
        self.perf_draw_every = max(1, int(perf_draw_base))
        self.base_steps_per_shot = self.cfg.steps_per_shot
        self.base_draw_every = self.cfg.draw_every

        # RNG for sampling
        self.rng = np.random.default_rng()
        self._gpu_rgba_buffer = None

        # movement-speed tuning (store baselines so slider changes keep swing tuning consistent)
        self._movement_slider_bounds = (2.0, 25.0)
        self._base_kmin_frac = float(getattr(cfg, 'kmin_frac', 0.15))
        self._base_kmax_frac = float(getattr(cfg, 'kmax_frac', 0.90))
        self._base_swing_power_scale = float(getattr(cfg, 'swing_power_scale', 0.05))
        self._base_impact_min_speed = float(getattr(cfg, 'impact_min_speed', 20.0))
        # direct multiplier applied to post-shot motion
        self._movement_speed_factor = float(max(self._movement_slider_bounds[0], getattr(cfg, 'movement_speed_scale', 17.5)))
        self._apply_movement_speed_tuning(initial=True)
        self._init_performance_helpers()

        self._log_startup_display_info()

        # mode management
        self._base_plot_ball = bool(self.cfg.PlotBall)
        self._base_plot_wave = bool(self.cfg.PlotWavePackage)
        self._mode_cycle = ["classical", "quantum", "mixed"]
        self._mode_styles = {
            "classical": ("C", "#1f77ff"),
            "quantum": ("Q", "#b066ff"),
            "mixed": ("X", "white"),
        }
        self._mode_index = self._mode_cycle.index("mixed")
        self.mode = self._mode_cycle[self._mode_index]

        self._apply_mode_settings(initial=True)

        # connect events
        self._connect_events()

        # Custom course management
        self._saved_state = None
        self._active_course = None
        self._course_stage = None
        self._course_trigger_keys = {
            'quantum_demo': '#',
            'advanced_showcase': '-',
        }
        self._course_presets = {
            'quantum_demo': [
                {
                    'map': 'single_wall',
                    'mode': 'classical',
                    'wave_profile': 'packet',
                    'shot_stop_mode': 'time',
                    'edge_boundary': 'reflect',
                    'single_wall_thickness_factor': 1.0,
                    'description': 'Stage 1: classical wall - observe purely classical rebound.',
                },
                {
                    'map': 'single_wall',
                    'mode': 'quantum',
                    'wave_profile': 'packet',
                    'shot_stop_mode': 'time',
                    'edge_boundary': 'reflect',
                    'single_wall_thickness_factor': 1.0,
                    'description': 'Stage 2: quantum wall - watch tunneling through the barrier.',
                },
                {
                    'map': 'double_slit',
                    'mode': 'mixed',
                    'wave_profile': 'front',
                    'shot_stop_mode': 'time',
                    'edge_boundary': 'reflect',
                    'description': 'Stage 3: double slit mixed mode - compare wave and ball outcomes.',
                },
            ],
            'advanced_showcase': [
                {
                    'map': 'single_wall',
                    'mode': 'mixed',
                    'wave_profile': 'front',
                    'shot_stop_mode': 'friction',
                    'edge_boundary': 'reflect',
                    'single_wall_thickness_factor': 2.5,
                    'description': 'Stage 1: adjustable wall - tweak thickness to study tunneling.',
                },
                {
                    'map': 'single_slit',
                    'mode': 'quantum',
                    'wave_profile': 'front',
                    'shot_stop_mode': 'time',
                    'edge_boundary': 'reflect',
                    'description': 'Stage 2: single-slit diffraction - pure wave behaviour.',
                },
                {
                    'map': 'no_obstacle',
                    'mode': 'quantum',
                    'wave_profile': 'front',
                    'shot_stop_mode': 'friction',
                    'edge_boundary': 'absorb',
                    'description': 'Stage 3: free propagation with absorbing edges - dissipate energy softly.',
                },
            ],
        }

        # Config panel state
        self._config_panel_active = False
        self._panel_axes_list: list = []
        self._panel_sliders: dict = {}
        self._panel_elements: dict = {}
        self._panel_updating = False
        self._panel_fig = None
        self._panel_close_cid = None
        self._panel_key_cid = None

        if getattr(self.cfg, 'show_control_panel', True):
            if Slider is None:
                print('[control panel] Matplotlib Slider widgets unavailable; set show_control_panel=False to suppress this message.')
            else:
                self._activate_config_panel()

    # ----- UI strings

    def _title_text(self):
        label = {
            'double_slit': 'double slit',
            'single_slit': 'single slit',
            'single_wall': 'wall',
            'Uni_Logo': 'Uni logo',
            'no_obstacle': 'no obstacle'
        }.get(self.cfg.map_kind, self.cfg.map_kind)
        return f"Quantum Mini-Golf - map: {label}"

    # ----- events
    def _connect_events(self):
        fig = self.viz.fig
        fig.canvas.mpl_connect('button_press_event', lambda e: None)
        fig.canvas.mpl_connect('button_release_event', lambda e: None)
        fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ----- mode helpers
    def _mode_allows_classical(self):
        return self._base_plot_ball and self.mode in ("classical", "mixed")

    def _mode_allows_quantum(self):
        return self._base_plot_wave and self.mode in ("quantum", "mixed")

    def _mode_allows_measure(self):
        return self._mode_allows_quantum() and self.cfg.quantum_measure

    def _apply_movement_speed_tuning(self, *, initial=False):
        bounds_min, bounds_max = self._movement_slider_bounds
        raw_value = float(getattr(self.cfg, 'movement_speed_scale', 17.5))
        if not math.isfinite(raw_value):
            raw_value = 8.0
        clamped_value = float(min(max(raw_value, bounds_min), bounds_max))
        if clamped_value != raw_value:
            self.cfg.movement_speed_scale = clamped_value
        self._movement_speed_factor = clamped_value

        # Reset swing tuning back to baseline so other systems stay consistent.
        self.cfg.kmax_frac = float(np.clip(self._base_kmax_frac, 0.05, 0.985))
        self.cfg.kmin_frac = float(np.clip(self._base_kmin_frac, 0.02, self.cfg.kmax_frac * 0.95))
        self.cfg.swing_power_scale = float(self._base_swing_power_scale)
        self.cfg.impact_min_speed = float(self._base_impact_min_speed)

        if getattr(self, '_config_panel_active', False) and not initial:
            self._refresh_slider_texts(draw=False)

    def _apply_tracker_size_scale(self):
        prev_scale = float(getattr(self, '_tracker_size_scale', 0.3))
        raw_factor = float(getattr(self.cfg, 'tracker_length_scale', 0.3))
        if not math.isfinite(raw_factor):
            raw_factor = 0.3
        factor = float(np.clip(raw_factor, 0.1, 1.5))
        self.cfg.tracker_length_scale = factor
        self.cfg.tracker_thickness_scale = factor
        self._tracker_size_scale = factor
        if self.tracker_cfg is None:
            return
        if self._tracker_base_length_px is None:
            self._tracker_base_length_px = float(getattr(self.tracker_cfg, 'putter_length_px', 380.0))
        if self._tracker_base_thickness_px is None:
            self._tracker_base_thickness_px = float(getattr(self.tracker_cfg, 'putter_thickness_px', 90.0))
        base_len = float(self._tracker_base_length_px or 0.0)
        base_thick = float(self._tracker_base_thickness_px or 0.0)
        self.tracker_cfg.putter_length_px = max(1.0, base_len * factor)
        self.tracker_cfg.putter_thickness_px = max(1.0, base_thick * factor)

    def _abort_current_shot(self):
        if self.shot_in_progress:
            self._abort_shot_requested = True
            self.shot_in_progress = False
        else:
            self._abort_shot_requested = False
        if self.tracker:
            self.tracker.pop_hits()

    def _ensure_gpu_rgba_buffer(self, shape):
        if not (self.be.USE_GPU and self.cfg.flags.gpu_viz):
            return None
        buf = self._gpu_rgba_buffer
        if buf is None or tuple(buf.shape[:2]) != tuple(shape):
            xp = self.be.xp
            buf = xp.empty(shape + (4,), dtype=xp.uint8)
            self._gpu_rgba_buffer = buf
        return buf

    def _gpu_encode_rgba(self, frame_idx_gpu):
        buf_gpu = self._ensure_gpu_rgba_buffer(frame_idx_gpu.shape)
        if buf_gpu is None:
            return None
        xp = self.be.xp
        lut_gpu = self.viz.get_cmap_lut(xp)
        xp.take(lut_gpu, frame_idx_gpu, axis=0, out=buf_gpu)
        cpu_buf = self.viz.ensure_rgba_buffer(frame_idx_gpu.shape)
        return self.be.to_cpu(buf_gpu, out=cpu_buf)

    def _present_density_frame(self, dens_xp, plot_wave=True):
        gpu_viz = self.be.USE_GPU and self.cfg.flags.gpu_viz
        frame = prepare_frame(
            dens_xp,
            self.be.xp,
            gpu_viz,
            self.cfg.flags,
            self.cfg.smooth_passes,
            max(1, self.cfg.display_downsample_factor),
            self.be.to_cpu,
            encode_uint8=True,
            norm_min=self.viz.wave_vmin,
            norm_max=self.viz.wave_vmax,
            return_device_array=gpu_viz,
        )
        if gpu_viz and hasattr(frame, 'device'):
            rgba = self._gpu_encode_rgba(frame)
            if rgba is not None:
                self.viz.draw_frame(rgba, plot_wave=plot_wave, pre_encoded=True)
                self._notify_frame_listeners()
                return
        self.viz.draw_frame(frame, plot_wave=plot_wave)
        self._notify_frame_listeners()

    def add_frame_listener(self, callback: Callable[["QuantumMiniGolfGame"], None]):
        if callback in self._frame_listeners:
            return
        self._frame_listeners.append(callback)

    def remove_frame_listener(self, callback: Callable[["QuantumMiniGolfGame"], None]):
        try:
            self._frame_listeners.remove(callback)
        except ValueError:
            pass

    def _notify_frame_listeners(self):
        if not self._frame_listeners:
            return
        for cb in tuple(self._frame_listeners):
            try:
                cb(self)
            except Exception as exc:
                print(f"[frame-listener] callback error: {exc}")

    def _render_wave_density(self, density_cpu):
        if density_cpu is None:
            return
        dens_xp = self.be.to_xp(density_cpu.astype(np.float32, copy=False), np.float32)
        if self._wavefront_active and self._last_phase_cpu is not None:
            phase_xp = self.be.to_xp(self._last_phase_cpu.astype(np.float32, copy=False), np.float32)
            stripes = 0.5 * (self.be.xp.cos(phase_xp) + 1.0)
            dens_xp = dens_xp * stripes.astype(self.be.xp.float32)
        self._present_density_frame(dens_xp, plot_wave=True)

    def _draw_idle_wave_preview(self):
        sigma = float(self.cfg.sigma0)
        if sigma <= 0:
            sigma = 1.0
        X = np.arange(self.Nx, dtype=np.float32)
        Y = np.arange(self.Ny, dtype=np.float32)
        Xg, Yg = np.meshgrid(X, Y, indexing='xy')
        x0, y0 = self.ball_pos
        profile = self._wavefront_profile
        if profile == 'front':
            sigma_y = float(max(1e-3, getattr(self.cfg, 'wavefront_sigma_y', sigma * 2.0)))
            trans_len = float(max(1e-3, getattr(self.cfg, 'wavefront_transition_len', sigma)))
            sigma_forward = float(max(1e-3, getattr(self.cfg, 'wavefront_sigma_forward', trans_len)))
            dir_vec = self._wavefront_dir.astype(np.float32, copy=True)
            norm = float(np.linalg.norm(dir_vec))
            if norm < 1e-6:
                dir_vec = np.array([1.0, 0.0], dtype=np.float32)
            else:
                dir_vec /= norm
            perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
            s = (Xg - x0) * dir_vec[0] + (Yg - y0) * dir_vec[1]
            p = (Xg - x0) * perp[0] + (Yg - y0) * perp[1]
            offset = float(max(0.0, getattr(self.cfg, 'wavefront_start_offset', 0.0)))
            s_shift = s + offset
            s_pos = np.maximum(s_shift, 0.0)
            forward_gauss = np.exp(-(s_pos ** 2) /
                                   (2.0 * sigma_forward * sigma_forward)).astype(np.float32)
            transition = 0.5 * (1.0 + np.tanh(s_shift / trans_len)).astype(np.float32)
            envelope_dir = forward_gauss * transition
            envelope_perp = np.exp(-(p ** 2) /
                                   (2.0 * sigma_y * sigma_y)).astype(np.float32)
            amp = np.clip(envelope_dir * envelope_perp, 0.0, None)
            kmax = float(self.be.to_cpu(self.k_max))
            kmag = 0.5 * (self.cfg.kmin_frac + self.cfg.kmax_frac) * kmax
            stripes = 0.5 * (1.0 + np.cos(kmag * s_shift)).astype(np.float32)
            dens = (amp ** 2) * stripes
            dens /= dens.sum() + 1e-12
            self._render_wave_density(dens)
            return
        else:
            amp = np.exp(-((Xg - x0) ** 2 + (Yg - y0) ** 2) /
                         (2.0 * sigma * sigma)).astype(np.float32)
        dens = amp * amp
        dens /= dens.sum() + 1e-12
        self._render_wave_density(dens)

    def _set_mode(self, mode: str):
        if mode not in self._mode_cycle:
            return
        self._mode_index = self._mode_cycle.index(mode)
        self.mode = mode
        self._apply_mode_settings()

    def _build_display_density(self, dens_xp, psi_xp=None):
        if psi_xp is None or not self._wavefront_active:
            return dens_xp
        xp = self.be.xp
        phase_xp = xp.angle(psi_xp)
        stripes = 0.5 * (xp.cos(phase_xp) + 1.0)
        return dens_xp * stripes.astype(xp.float32)

    def _friction_speed_scale(self, step_idx: int, total_steps: int, coeffs: tuple[float, float, float]) -> float:
        if total_steps <= 1:
            return 0.0
        s = float(step_idx) / float(max(1, total_steps - 1))
        s = float(min(max(s, 0.0), 1.0))
        a, b, c = coeffs
        scale = 1.0 - (a * s + b * (s ** 2) + c * (s ** 3))
        return float(np.clip(scale, 0.0, 1.0))

    def _apply_mode_settings(self, initial=False):
        label, color = self._mode_styles[self.mode]
        if not self._minimal_annotations:
            self.viz.update_mode_label(label, color)
        show_ball = self._mode_allows_classical()
        show_wave = self._mode_allows_quantum()
        self.viz.set_ball_visible(show_ball)
        if not show_ball:
            self.viz.clear_classical_overlay()
        self.viz.set_wave_visible(show_wave)
        self.viz.clear_wave_overlay(show_wave)
        if show_wave:
            if self._last_density_cpu is not None:
                self._render_wave_density(self._last_density_cpu)
            else:
                self._draw_idle_wave_preview()
            if self.show_interference and self._interference_profile is not None:
                self.viz.set_interference_visible(True)
                self.viz.update_interference_pattern(self._interference_profile)
        else:
            self.viz.draw_frame(
                np.zeros((self.Ny, self.Nx), dtype=np.float32), plot_wave=False)
            self.show_info = False
            self.viz.set_info_visibility(False)
            self.viz.measure_point.set_visible(False)
            self.viz.measure_marker.set_visible(False)
            self.viz.set_wave_path_label(False)
            self.viz.show_messages(False, False)
            self.show_interference = False
            self._interference_profile = None
            self.viz.set_interference_visible(False)
        self.viz.set_wave_path_label(self.show_info and show_wave)
        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        else:
            self.viz.fig.canvas.draw_idle()

    def _cycle_mode(self):
        if self.shot_in_progress:
            return
        self._mode_index = (self._mode_index + 1) % len(self._mode_cycle)
        self.mode = self._mode_cycle[self._mode_index]
        self._apply_mode_settings()

    def _log_startup_display_info(self, force: bool = False) -> None:
        if not force and getattr(self, "_startup_info_logged", False):
            return
        self._startup_info_logged = True

        tracker_cfg = self.tracker_cfg
        camera_line = "tracker camera frame: unavailable"
        cfg_line = "tracker cfg: unavailable"
        if tracker_cfg is not None:
            frame_w = getattr(tracker_cfg, "frame_width", None)
            frame_h = getattr(tracker_cfg, "frame_height", None)
            if frame_w and frame_h:
                camera_line = f"tracker camera frame: {int(frame_w)} x {int(frame_h)} px"
            crop_vals = (
                getattr(tracker_cfg, "crop_x1", None),
                getattr(tracker_cfg, "crop_x2", None),
                getattr(tracker_cfg, "crop_y1", None),
                getattr(tracker_cfg, "crop_y2", None),
            )
            if all(v is not None for v in crop_vals):
                cx1, cx2, cy1, cy2 = (int(v) for v in crop_vals)  # type: ignore[arg-type]
                roi_w = max(0, cx2 - cx1)
                roi_h = max(0, cy2 - cy1)
                if roi_w > 0 and roi_h > 0:
                    cfg_line = f"tracker cfg: {roi_w} x {roi_h} px (ROI)"
                elif frame_w and frame_h:
                    cfg_line = f"tracker cfg: {int(frame_w)} x {int(frame_h)} px (full frame)"
            elif frame_w and frame_h:
                cfg_line = f"tracker cfg: {int(frame_w)} x {int(frame_h)} px (full frame)"

        try:
            canvas_w, canvas_h = self.viz.fig.canvas.get_width_height()
            course_line = f"course window: {int(canvas_w)} x {int(canvas_h)} px"
        except Exception:
            course_line = "course window: unavailable"

        screen_lines: list[str] = []
        if QtWidgets is not None:
            app = QtWidgets.QApplication.instance()  # type: ignore[attr-defined]
            if app is not None:
                try:
                    screens = app.screens()
                except Exception:
                    screens = []
                for idx, screen in enumerate(screens, start=1):
                    try:
                        geometry = screen.geometry()
                        width = int(getattr(geometry, "width")())
                        height = int(getattr(geometry, "height")())
                    except Exception:
                        try:
                            size = screen.size()
                            width = int(getattr(size, "width")())
                            height = int(getattr(size, "height")())
                        except Exception:
                            width = height = -1
                    if width > 0 and height > 0:
                        screen_lines.append(f"screen {idx}: {width} x {height} px")
        if not screen_lines:
            try:
                import tkinter as tk  # type: ignore
            except Exception:
                tk = None  # type: ignore
            if tk is not None:
                root = None
                try:
                    root = tk.Tk()
                    root.withdraw()
                    width = int(root.winfo_screenwidth())
                    height = int(root.winfo_screenheight())
                    screen_lines.append(f"screen 1: {width} x {height} px")
                except Exception:
                    pass
                finally:
                    if root is not None:
                        try:
                            root.destroy()
                        except Exception:
                            pass
        if not screen_lines:
            screen_lines.append("screen 1: unavailable")

        print(camera_line)
        print(course_line)
        for line in screen_lines:
            print(line)
        print(cfg_line)

    def _schedule_display_info_refresh(self) -> None:
        canvas = getattr(self.viz, "fig", None)
        if canvas is None:
            self._log_startup_display_info(force=True)
            return
        try:
            timer = self.viz.fig.canvas.new_timer(interval=150)
        except Exception:
            self._log_startup_display_info(force=True)
            return
        if hasattr(timer, "single_shot"):
            try:
                timer.single_shot = True  # type: ignore[attr-defined]
            except Exception:
                pass
        if self._display_info_timer is not None:
            try:
                self._display_info_timer.stop()
            except Exception:
                pass
        timer.add_callback(self._log_startup_display_info, True)
        self._display_info_timer = timer
        timer.start()

    def _ensure_exponents(self, dt: float, force: bool = False) -> None:
        dt = float(dt)
        if force or self._current_exp_dt is None or abs(self._current_exp_dt - dt) > self._dt_tolerance:
            self.course.update_exponents(dt, self.k2, self.c64)
            self._current_exp_dt = dt
            if self._perf_enabled:
                self._perf_expK_powers.clear()

    def _compute_phase_dt(self) -> float:
        denom = max(self._max_combined_potential, 0.5 * self._k2_cpu_max, 1e-6)
        return self._performance_theta / denom

    def _ensure_fft_plan(self, psi):
        if not (self._perf_enabled and self.be.USE_GPU and self.be.cp is not None):
            return None
        cp = self.be.cp
        key = (tuple(int(v) for v in psi.shape), str(psi.dtype))
        plan = self._perf_fft_plan_cache.get(key)
        if plan is None:
            try:
                plan = cp.fft.config.get_plan(psi)
                self._perf_fft_plan_cache[key] = plan
            except Exception as exc:
                if not self._perf_fft_plan_fail_logged:
                    print(f"[perf] Unable to cache cuFFT plan, falling back to default transforms: {exc}")
                    self._perf_fft_plan_fail_logged = True
                return None
        return plan

    def _get_expK_power(self, base_expK, m: int):
        val = self._perf_expK_powers.get(m)
        if val is None:
            xp = self.be.xp
            try:
                val = xp.power(base_expK, m).astype(self.c64, copy=False)
            except Exception:
                val = xp.power(base_expK, m)
            self._perf_expK_powers[m] = val
        return val

    def _build_friction_lookup(self, base_expV_half, base_expK, dt_eff, xp):
        bins = int(max(2, getattr(self.cfg, 'performance_friction_bins', 32)))
        scales = np.linspace(0.0, 1.0, bins, dtype=np.float32)
        expV_list = []
        expK_list = []
        V_operator = getattr(self.course, 'V_operator', None)
        if V_operator is None:
            self._perf_friction_expV = None
            self._perf_friction_expK = None
            self._perf_friction_scales = None
            return
        for s in scales:
            if s >= 1.0 - 1e-6:
                expV_list.append(base_expV_half)
                expK_list.append(base_expK)
            elif s <= 1e-6:
                expV_list.append(xp.ones_like(base_expV_half, dtype=self.c64))
                expK_list.append(xp.ones_like(base_expK, dtype=self.c64))
            else:
                half_dt = dt_eff * float(s) * 0.5
                expV = xp.exp(V_operator * half_dt).astype(self.c64, copy=False)
                expK_scaled = xp.exp((-1j * self.k2 * dt_eff * float(s) / 2.0).astype(self.c64)).astype(self.c64, copy=False)
                expV_list.append(expV)
                expK_list.append(expK_scaled)
        self._perf_friction_expV = expV_list
        self._perf_friction_expK = expK_list
        self._perf_friction_scales = scales

    def _lookup_friction_index(self, scale: float) -> int:
        if self._perf_friction_scales is None:
            return 0
        bins = len(self._perf_friction_scales)
        if bins <= 1:
            return 0
        idx = int(round(float(scale) * (bins - 1)))
        return min(max(idx, 0), bins - 1)

    def _maybe_burst_drift(self, psi, base_expK, dt_eff, steps_remaining, plan):
        if not self._perf_enabled or self._perf_drift_threshold <= 0.0:
            return psi, 1
        if self._obstacle_mask is None:
            if not self._perf_drift_fail_logged:
                print("[perf] Obstacle mask unavailable; skipping burst drift optimisation.")
                self._perf_drift_fail_logged = True
            return psi, 1
        xp = self.be.xp
        dens = xp.abs(psi) ** 2
        overlap = xp.sum(dens * self._obstacle_mask)
        overlap_val = float(self.be.to_cpu(overlap))
        if overlap_val >= self._perf_drift_threshold:
            return psi, 1
        m = min(self._perf_max_drift_steps, steps_remaining)
        if m <= 1:
            return psi, 1
        expK_power = self._get_expK_power(base_expK, m)
        if plan is not None and self.be.USE_GPU:
            with plan:
                psi_k = self.be.fft2(psi)
                psi_k *= expK_power
                psi = self.be.ifft2(psi_k)
        else:
            psi_k = self.be.fft2(psi)
            psi_k *= expK_power
            psi = self.be.ifft2(psi_k)
        psi = psi.astype(self.c64, copy=False)
        return psi, m

    def _init_performance_helpers(self) -> None:
        if not self._perf_enabled:
            return
        if self._perf_window_enabled:
            print("[perf] moving window optimisation not yet available; using full-grid evolution.")
            self._perf_window_enabled = False
        if getattr(self.cfg, 'fast_mode_paraxial', False) and not self._perf_fast_mode_logged:
            print("[perf] fast-mode-paraxial not implemented; falling back to split-step propagation.")
            self._perf_fast_mode_logged = True
            self.cfg.fast_mode_paraxial = False

    def _start_tracker_poll(self):
        if not self.tracker:
            return
        self._update_tracker_reference()
        self._tracker_timer = self.viz.fig.canvas.new_timer(interval=33)
        self._tracker_timer.add_callback(self._poll_tracker)
        self._tracker_timer.start()
        self._refresh_tracker_overlay_mode()

    def _compute_tracker_decoupled(self) -> bool:
        if not getattr(self.cfg, 'decouple_tracker_overlay', False):
            return False
        if not getattr(self.cfg, 'use_tracker', False):
            return False
        if self.tracker is None:
            return False
        if getattr(self.cfg, 'enable_mouse_swing', False):
            return False
        if self._tracker_force_disabled:
            return False
        return True

    def _refresh_tracker_overlay_mode(self) -> None:
        decoupled = self._compute_tracker_decoupled()
        if decoupled != self._tracker_decoupled:
            self._tracker_decoupled = decoupled
            if not decoupled:
                self._tracker_overlay_enabled = bool(self._tracker_overlay_initial)
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)

    def _on_close(self, _event):
        if self._tracker_timer is not None:
            try:
                self._tracker_timer.stop()
            except Exception:
                pass
            self._tracker_timer = None
        self._stop_debug_logging()
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
        if self._config_panel_active:
            self._deactivate_config_panel()
        self._stop_playback()

    def _update_tracker_reference(self):
        if not self.tracker:
            return
        self.tracker.update_reference_point(
            tuple(self.ball_pos), (self.Nx, self.Ny))

    def _tracker_px_to_game(self, px: tuple[float, float]) -> tuple[float, float]:
        if not self.tracker_cfg:
            return float(px[0]), float(px[1])
        calibration = getattr(self.tracker_cfg, "calibration", None)
        if calibration is None:
            x = px[0] / self.tracker_cfg.frame_width * self.Nx
            y = (1.0 - px[1] / self.tracker_cfg.frame_height) * self.Ny
            return self._apply_tracker_correction(float(x), float(y))
        board_x_raw, board_y_raw = calibration.camera_to_board(px)
        board_x = float(board_x_raw)
        board_y = float(board_y_raw)
        scale_x = self.Nx / max(calibration.board_width, 1e-6)
        scale_y = self.Ny / max(calibration.board_height, 1e-6)
        game_x = float(board_x * scale_x)
        game_y = float((calibration.board_height - board_y) * scale_y)
        return self._apply_tracker_correction(game_x, game_y)

    def _tracker_dir_to_game(
        self,
        origin_px: tuple[float, float],
        direction_px: tuple[float, float],
    ) -> np.ndarray:
        if not self.tracker_cfg:
            return np.array(direction_px, dtype=float)
        calibration = getattr(self.tracker_cfg, "calibration", None)
        if calibration is None:
            scale_x = self.Nx / self.tracker_cfg.frame_width
            scale_y = self.Ny / self.tracker_cfg.frame_height
            return np.array([direction_px[0] * scale_x, -direction_px[1] * scale_y], dtype=float)

        origin_board_raw = calibration.camera_to_board(origin_px)
        tip_px = (origin_px[0] + direction_px[0], origin_px[1] + direction_px[1])
        tip_board_raw = calibration.camera_to_board(tip_px)
        delta_board = np.array(
            [
                float(tip_board_raw[0] - origin_board_raw[0]),
                float(tip_board_raw[1] - origin_board_raw[1]),
            ],
            dtype=float,
        )
        scale_x = self.Nx / max(calibration.board_width, 1e-6)
        scale_y = self.Ny / max(calibration.board_height, 1e-6)
        return np.array(
            [delta_board[0] * scale_x, -delta_board[1] * scale_y],
            dtype=float,
        )

    def _apply_tracker_correction(self, x: float, y: float) -> tuple[float, float]:
        if getattr(self, '_tracker_auto_scale', False):
            x = x * self._tracker_corr_scale_x + self._tracker_corr_offset_x
            y = y * self._tracker_corr_scale_y + self._tracker_corr_offset_y
        calibration_present = bool(getattr(self.tracker_cfg, "calibration", None))
        if calibration_present:
            margin = float(max(0.0, getattr(self.cfg, 'tracker_coord_margin', 6.0)))
            max_x = float(self.Nx + margin)
            max_y = float(self.Ny + margin)
            min_x = float(-margin)
            min_y = float(-margin)
        else:
            margin = 0.0
            max_x = float(self.Nx)
            max_y = float(self.Ny)
            min_x = 0.0
            min_y = 0.0
        x = float(min(max(x, min_x), max_x))
        y = float(min(max(y, min_y), max_y))
        return x, y

    def _register_tracker_span(self, led_a_game: tuple[float, float], led_b_game: tuple[float, float]) -> None:
        if not self._tracker_auto_scale:
            return
        ax, ay = float(led_a_game[0]), float(led_a_game[1])
        bx, by = float(led_b_game[0]), float(led_b_game[1])
        min_x = min(ax, bx)
        max_x = max(ax, bx)
        min_y = min(ay, by)
        max_y = max(ay, by)
        if 0.0 <= min_x <= float(self.Nx):
            if self._tracker_min_x is None or min_x < self._tracker_min_x:
                self._tracker_min_x = min_x
        if 0.0 <= max_x <= float(self.Nx):
            if self._tracker_max_x is None or max_x > self._tracker_max_x:
                self._tracker_max_x = max_x
        if 0.0 <= min_y <= float(self.Ny):
            if self._tracker_min_y is None or min_y < self._tracker_min_y:
                self._tracker_min_y = min_y
        if 0.0 <= max_y <= float(self.Ny):
            if self._tracker_max_y is None or max_y > self._tracker_max_y:
                self._tracker_max_y = max_y
        self._update_tracker_scale()

    def _update_tracker_scale(self) -> None:
        if not self._tracker_auto_scale:
            return
        target_min_x = 1.0
        target_max_x = float(self.Nx - 1)
        min_x = self._tracker_min_x
        max_x = self._tracker_max_x
        if min_x is not None and max_x is not None:
            span = max_x - min_x
            target_span = target_max_x - target_min_x
            if span >= target_span * 0.6:
                scale = target_span / max(span, 1e-6)
                offset = target_min_x - min_x * scale
                alpha = 0.15
                self._tracker_corr_scale_x = (1.0 - alpha) * self._tracker_corr_scale_x + alpha * scale
                self._tracker_corr_offset_x = (1.0 - alpha) * self._tracker_corr_offset_x + alpha * offset
        target_min_y = 1.0
        target_max_y = float(self.Ny - 1)
        min_y = self._tracker_min_y
        max_y = self._tracker_max_y
        if min_y is not None and max_y is not None:
            span_y = max_y - min_y
            target_span_y = target_max_y - target_min_y
            if span_y >= target_span_y * 0.6:
                scale_y = target_span_y / max(span_y, 1e-6)
                offset_y = target_min_y - min_y * scale_y
                alpha = 0.15
                self._tracker_corr_scale_y = (1.0 - alpha) * self._tracker_corr_scale_y + alpha * scale_y
                self._tracker_corr_offset_y = (1.0 - alpha) * self._tracker_corr_offset_y + alpha * offset_y

    def _poll_tracker(self):
        if not self.tracker:
            return
        state = self.tracker.get_state()
        self._update_tracker_reference()
        span_px = float(state.span_px or 0.0)
        min_span = float(getattr(self.cfg, 'tracker_min_span_px', 0.0))
        center_px = state.center_px
        dir_px = state.direction_px
        led_a_px = getattr(state, 'led_a_px', None)
        led_b_px = getattr(state, 'led_b_px', None)
        max_span_px = float(getattr(self.cfg, 'tracker_max_span_px', 220.0))
        visible = (
            state.visible
            and center_px is not None
            and dir_px is not None
            and led_a_px is not None
            and led_b_px is not None
            and span_px >= max(min_span, 1e-6)
            and span_px <= max_span_px
        )
        geometry_ok = False
        angle_deg = 0.0
        center = (0.0, 0.0)
        length_game = 0.0
        thickness_game = 0.0
        area_game = 0.0
        if visible:
            dir_step_game = self._tracker_dir_to_game(center_px, dir_px)
            step_norm = np.linalg.norm(dir_step_game)
            led_a_game = self._tracker_px_to_game(led_a_px)
            led_b_game = self._tracker_px_to_game(led_b_px)
            delta_game = np.array(
                [led_b_game[0] - led_a_game[0], led_b_game[1] - led_a_game[1]],
                dtype=float,
            )
            length_raw = float(np.linalg.norm(delta_game))
            if length_raw >= 1e-6:
                center = (
                    (led_a_game[0] + led_b_game[0]) * 0.5,
                    (led_a_game[1] + led_b_game[1]) * 0.5,
                )
                dir_unit = delta_game / max(length_raw, 1e-6)
                angle_deg = math.degrees(math.atan2(dir_unit[1], dir_unit[0]))
                length_scale = float(getattr(self.cfg, 'tracker_length_scale', 1.0))
                length_game = length_raw * length_scale
                if length_game <= 0.0:
                    length_game = length_raw
                overlay_thickness_px = max(1.0, float(getattr(self.cfg, 'tracker_overlay_thickness_px', 4.0)))
                perp_cam = (-dir_px[1], dir_px[0])
                perp_step_game = self._tracker_dir_to_game(center_px, perp_cam)
                perp_norm = np.linalg.norm(perp_step_game)
                if perp_norm < 1e-6:
                    perp_norm = step_norm if step_norm >= 1e-6 else length_raw / max(span_px, 1e-6)
                thickness_scale = float(getattr(self.cfg, 'tracker_thickness_scale', 1.0))
                thickness_game = float(perp_norm * overlay_thickness_px * thickness_scale)
                min_thickness = float(perp_norm * max(1.5, overlay_thickness_px * 0.25))
                if thickness_game < min_thickness:
                    thickness_game = min_thickness
                area_game = float(max(0.0, length_game * thickness_game))
                geometry_ok = True
                self._register_tracker_span(led_a_game, led_b_game)
        self._tracker_last_area = float(area_game)
        allow_hits = geometry_ok and not self._tracker_force_disabled
        self._tracker_area_valid = allow_hits
        overlay_any_visible = geometry_ok and not self._tracker_force_disabled
        overlay_visible = overlay_any_visible and self._tracker_overlay_enabled
        if geometry_ok:
            self.viz.update_putter_overlay(
                center, length_game, thickness_game, angle_deg, overlay_visible
            )
        else:
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)
        hits = self.tracker.pop_hits()
        if allow_hits:
            for hit in hits:
                self._handle_tracker_hit(hit)

    def _handle_tracker_hit(self, hit):
        if self.shot_in_progress or self.game_over:
            return
        if not self._tracker_area_valid:
            return
        dir_game = self._tracker_dir_to_game(hit.center_px, hit.direction_px)
        norm = np.linalg.norm(dir_game)
        if norm < 1e-6:
            return
        dir_unit = dir_game / norm
        raw_speed = float(hit.speed_px_s)
        scaled_speed = raw_speed * float(self.cfg.tracker_speed_scale)
        speed = max(self.cfg.impact_min_speed, scaled_speed)
        kvec = self._compute_kvec_from_swing(
            dir_unit.astype(np.float32), speed)
        if kvec is None:
            return
        self._shoot(kvec)

    def _inside_axes(self, e):
        return e.inaxes is self.viz.ax and (0 <= e.xdata <= self.Nx) and (0 <= e.ydata <= self.Ny)

    def _event_to_data(self, e):
        try:
            x, y = self.viz.ax.transData.inverted().transform((e.x, e.y))
            return np.array([x, y], dtype=float)
        except Exception:
            return None

    def _on_motion(self, e):
        allow_mouse = bool(getattr(self.cfg, 'enable_mouse_swing', False))

        if self.tracker and not allow_mouse:
            self.cursor_inside = False
            self.viz.indicator_patch.set_visible(False)
            self.last_mouse_pos = None
            self.last_mouse_t = None
            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()
            return

        if self.shot_in_progress or self.game_over:
            return

        if not allow_mouse:
            self.cursor_inside = False
            self.viz.indicator_patch.set_visible(False)
            self.last_mouse_pos = None
            self.last_mouse_t = None
            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()
            return

        # debounce
        if self.cfg.flags.event_debounce:
            now_local = time.perf_counter()
            if self.last_mouse_t is not None and (now_local - self.last_mouse_t) < (self.cfg.debounce_ms / 1000.0):
                return

        p = self._event_to_data(e)
        if p is None:
            return

        if not self._inside_axes(e):
            self.cursor_inside = False
            self.viz.indicator_patch.set_visible(False)
            self.last_mouse_pos = None
            self.last_mouse_t = None
            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()
            return

        # show indicator
        self.cursor_inside = True
        self.indicator_pos = p
        self.viz.indicator_patch.center = (p[0], p[1])
        self.viz.indicator_patch.set_visible(True)

        prev_pos = self.last_mouse_pos
        prev_t = self.last_mouse_t
        v_vec = None
        dt = None
        now = time.perf_counter()
        if prev_pos is not None and prev_t is not None:
            dt = max(1e-4, now - prev_t)
            v_vec = (p - prev_pos) / dt
        self.last_mouse_pos = p
        self.last_mouse_t = now

        if v_vec is not None:
            self._check_collision_and_shoot(p, v_vec, prev_pos, dt)

        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        else:
            self.viz.fig.canvas.draw_idle()

    def _on_key(self, e):
        key = (e.key or "").lower()
        if key == 'q':
            plt.close(self.viz.fig)
        elif key == 'f':
            self._log_startup_display_info(force=True)
            self._schedule_display_info_refresh()
        elif key == 'r':
            self._abort_current_shot()
            restored = self._restore_normal_state()
            if restored:
                self._reset()
                self._abort_shot_requested = False
            else:
                self._double_reset()
        elif key == 'tab':
            self._toggle_map()
        elif key == 'c':
            self._cycle_mode()
        elif key == 'm':
            if self._mode_allows_measure():
                self._measure_now()
        elif key == 'i':
            if not self._mode_allows_quantum():
                return
            if self.game_over and not self.shot_in_progress:
                if not self.show_info:
                    self._toggle_info_overlay()
                else:
                    if self._mode_allows_measure():
                        self._measure_now()
                    else:
                        print('[i] Quantum measurements are disabled in this mode.')
            else:
                self._toggle_info_overlay()
        elif key == '#':
            self._cycle_course('quantum_demo')
        elif key == '-':
            self._cycle_course('advanced_showcase')
        elif key == 'b':
            self._toggle_edge_boundary()
        elif key == 'w':
            self._toggle_wave_profile()
        elif key == 't':
            self._toggle_shot_stop_mode()
        elif key == 'g':
            self._toggle_mouse_swing()
        elif key == 'o':
            self._toggle_tracker_overlay()
        elif key == 'u':
            self._toggle_config_panel()
        elif key == 'p':
            self._cycle_background()
            toolbar = getattr(e.canvas, 'toolbar', None)
            if toolbar is not None and getattr(toolbar, 'mode', '') == 'pan/zoom':
                try:
                    toolbar.pan()
                except Exception:
                    try:
                        toolbar.mode = ''
                    except Exception:
                        pass
        elif key == 'l':
            self._toggle_interference_pattern()
        elif key == 'd':
            self._play_recording()
        elif key == 'h':
            self._print_hotkey_help()

    def _toggle_map(self):
        if self.shot_in_progress:
            return
        order = ['double_slit', 'single_slit', 'single_wall', 'Uni_Logo', 'no_obstacle']
        try:
            idx = order.index(self.cfg.map_kind)
        except ValueError:
            idx = 0
        self._switch_map(order[(idx + 1) % len(order)])

    def _switch_map(self, kind):
        if self.shot_in_progress:
            return
        self.course.set_map(kind)
        self._refresh_course_assets()
        self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
        self.viz.set_course_patches(self.course.course_patches)
        self._reset(ball_only=False)
        self.viz.update_title(self._title_text())

    def _toggle_info_overlay(self):
        if not self._mode_allows_quantum():
            return
        shot_guard = self.shot_in_progress and not self._overlay_compute_during_shot
        if shot_guard:
            if not self.show_info:
                self.show_info = True
                self._pending_info_overlay = True
                self.viz.set_info_visibility(False)
                self.viz.measure_point.set_visible(False)
                self.viz.measure_marker.set_visible(False)
                self.viz.set_wave_path_label(False)
            else:
                self.show_info = False
                self._pending_info_overlay = False
                self.viz.set_info_visibility(False)
                self.viz.set_wave_path_label(False)
                self.viz.show_messages(False, False)
                if self.cfg.flags.blitting:
                    self.viz._blit_draw()
                else:
                    self.viz.fig.canvas.draw_idle()
            return

        self.show_info = not self.show_info
        if not self.show_info:
            self._pending_info_overlay = False
            self.viz.set_info_visibility(False)
            self.viz.set_wave_path_label(False)
            self.viz.show_messages(False, False)
            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()
            return

        self._pending_info_overlay = False
        if self._last_density_cpu is not None and self._mode_allows_quantum():
            dens_xp = self.be.to_xp(self._last_density_cpu, np.float32)
            xp = self.be.xp
            ex, ey, a1, b1, ang = covariance_ellipse(
                self.Xgrid, self.Ygrid, dens_xp, xp, self.be.to_cpu
            )
            self.viz.update_overlay_from_stats(ex, ey, a1, b1, ang, show=True)
            if self._last_measure_xy is not None:
                mx, my = self._last_measure_xy
                self.viz.set_measure_point(mx, my, True)
        self.viz.set_wave_path_label(True)
        self.viz.hole_msg.set_visible(False)
        self.viz.hole_msg_ball.set_visible(False)
        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        else:
            self.viz.fig.canvas.draw_idle()

    def _toggle_interference_pattern(self):
        if not self._mode_allows_quantum():
            print('[l] Interference view requires a quantum-enabled mode (quantum or mixed).')
            return
        if self.shot_in_progress:
            print('[l] Wait for the current shot to finish before toggling the interference view.')
            return
        if self.show_interference:
            self.show_interference = False
            self.viz.set_interference_visible(False)
            return
        has_profile = self._update_interference_pattern(force=True)
        if has_profile:
            self.show_interference = True
            print('[l] Showing interference profile along the exit wall (press l to hide).')
        else:
            self.viz.set_interference_visible(False)
            print('[l] No interference data yet. Take a shot to generate the pattern.')

    def _update_interference_pattern(self, density=None, force=False):
        if density is None:
            density = self._last_density_cpu
        if density is None:
            return False
        try:
            profile = self._compute_interference_profile(density)
        except Exception as exc:
            print(f"[interference] failed to compute profile: {exc}")
            return False
        self._interference_profile = profile
        if self.show_interference or force:
            self.viz.set_interference_visible(True)
            self.viz.update_interference_pattern(profile)
        return True

    def _refresh_course_assets(self, *, update_visuals: bool = True) -> None:
        """
        Synchronise cached course-dependent data after the layout changes.
        """
        self.hole_center = self.course.hole_center.copy()
        self._max_combined_potential = float(
            getattr(self.course, 'max_potential', 0.0) + getattr(self.course, 'max_absorber', 0.0)
        )
        self._obstacle_mask = getattr(self.course, 'obstacle_mask', None)
        self._obstacle_mask_cpu = getattr(self.course, 'obstacle_mask_cpu', None)
        if update_visuals and hasattr(self, 'viz') and self.viz is not None:
            try:
                self.viz.set_hole_geometry(self.hole_center, self.cfg.hole_r)
            except Exception:
                pass

    def _init_tracker(self, enabled: bool) -> None:
        if not enabled:
            return

        preloaded_calibration = getattr(self.cfg, 'tracker_calibration_data', None)
        calibration = preloaded_calibration if isinstance(preloaded_calibration, CalibrationData) else None
        if calibration is None:
            calib_path_raw = getattr(self.cfg, 'tracker_calibration_path', None)
            candidate_paths: list[Path] = []
            if calib_path_raw:
                candidate_paths.append(Path(calib_path_raw))
            else:
                candidate_paths.extend(
                    [
                        Path("calibration") / "course_calibration.pkl",
                        Path("calibration") / "course_calibration.json",
                    ]
                )
            for candidate in candidate_paths:
                try:
                    if candidate.exists():
                        calibration = CalibrationData.load(candidate)
                        print(f"Tracker calibration loaded from {candidate}")
                        break
                except Exception as exc:
                    print(f"Tracker calibration load failed ({candidate}): {exc}")

        tracker_kwargs = dict(
            show_debug_window=self.cfg.tracker_debug_window,
            crop_x1=getattr(self.cfg, 'tracker_crop_x1', None),
            crop_x2=getattr(self.cfg, 'tracker_crop_x2', None),
            crop_y1=getattr(self.cfg, 'tracker_crop_y1', None),
            crop_y2=getattr(self.cfg, 'tracker_crop_y2', None),
            threshold=int(getattr(self.cfg, 'tracker_threshold', 55)),
        )
        if calibration is not None:
            tracker_kwargs["frame_width"] = calibration.frame_width
            tracker_kwargs["frame_height"] = calibration.frame_height
            tracker_kwargs["calibration"] = calibration

        self.tracker_cfg = TrackerConfig(**tracker_kwargs)
        self._tracker_base_length_px = float(getattr(self.tracker_cfg, 'putter_length_px', 380.0))
        self._tracker_base_thickness_px = float(getattr(self.tracker_cfg, 'putter_thickness_px', 90.0))
        self._apply_tracker_size_scale()
        try:
            self.cfg.tracker_threshold = int(self.tracker_cfg.threshold)
        except Exception:
            pass

        try:
            self.tracker = TrackerManager(self.tracker_cfg)
            self.tracker.start()
        except Exception as exc:
            print(f'Tracker disabled: {exc}')
            self.tracker = None
        else:
            self._start_tracker_poll()
            if self._debug_log_enabled:
                self._start_debug_logging()

    # ----- multi-shot helpers
    def _update_shot_counter(self):
        if not getattr(self, '_multiple_shots_enabled', False):
            self.viz.update_shot_counter(None)
        else:
            self.viz.update_shot_counter(self._shot_count)

    def _reset_shot_counter(self):
        if not getattr(self, '_multiple_shots_enabled', False):
            self.viz.update_shot_counter(None)
            return
        self._shot_count = 0
        self._update_shot_counter()

    # ----- debug logging
    def _start_debug_logging(self):
        if not self._debug_log_enabled or not getattr(self.cfg, 'use_tracker', False):
            return
        if self.tracker is None or self.viz is None:
            return
        if self._debug_log_timer is not None:
            return
        try:
            self._debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if not self._debug_session_started:
            start_payload = {
                "event": "debug_log_start",
                "timestamp_wall": time.time(),
                "tracker_frame": {
                    "width": int(getattr(self.tracker_cfg, 'frame_width', 0)) if self.tracker_cfg else None,
                    "height": int(getattr(self.tracker_cfg, 'frame_height', 0)) if self.tracker_cfg else None,
                },
            }
            self._write_debug_log(start_payload)
            self._debug_session_started = True
        self._debug_prev_putter = {"center": None, "time": None}
        self._debug_prev_tracker = {"center": None, "time": None}
        timer = self.viz.fig.canvas.new_timer(interval=100)
        timer.add_callback(self._debug_log_tick)
        self._debug_log_timer = timer
        try:
            timer.start()
        except Exception:
            self._debug_log_timer = None

    def _stop_debug_logging(self):
        timer = self._debug_log_timer
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
            self._debug_log_timer = None
        if self._debug_session_started:
            self._write_debug_log({"event": "debug_log_stop", "timestamp_wall": time.time()})
            self._debug_session_started = False

    def _write_debug_log(self, payload: dict):
        if self._debug_log_write_failed:
            return
        try:
            with self._debug_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
        except Exception as exc:
            print(f"[debug-log] Failed to write tracker debug entry: {exc}")
            self._debug_log_write_failed = True

    def _debug_log_tick(self):
        if not getattr(self.cfg, 'use_tracker', False) or self.tracker is None:
            return
        try:
            state = self.tracker.get_state()
        except Exception as exc:
            if not self._debug_log_write_failed:
                print(f"[debug-log] Tracker state unavailable: {exc}")
                self._debug_log_write_failed = True
            return
        if not state.timestamp:
            return

        frame_width = int(getattr(self.tracker_cfg, 'frame_width', 0)) if self.tracker_cfg else None
        frame_height = int(getattr(self.tracker_cfg, 'frame_height', 0)) if self.tracker_cfg else None

        def _tuple_to_list(value):
            if value is None:
                return None
            return [float(value[0]), float(value[1])]

        try:
            reference_px = self.tracker.get_reference_point()
        except Exception:
            reference_px = None

        putter_state = self.viz.get_putter_state()
        putter_visible = bool(putter_state.get("visible")) if putter_state else False
        putter_center = putter_state.get("center") if putter_state else None
        now_perf = time.perf_counter()
        putter_speed = None
        prev_putter_center = self._debug_prev_putter.get("center")
        prev_putter_time = self._debug_prev_putter.get("time")
        if putter_visible and putter_center is not None and prev_putter_center is not None and prev_putter_time is not None:
            dt_putter = now_perf - float(prev_putter_time)
            if dt_putter > 1e-6:
                dx = float(putter_center[0]) - float(prev_putter_center[0])
                dy = float(putter_center[1]) - float(prev_putter_center[1])
                putter_speed = math.hypot(dx, dy) / dt_putter
        if putter_visible and putter_center is not None:
            self._debug_prev_putter = {"center": (float(putter_center[0]), float(putter_center[1])), "time": now_perf}
        else:
            self._debug_prev_putter = {"center": None, "time": now_perf}

        tracker_visible = bool(
            state.visible
            and state.center_px is not None
            and state.direction_px is not None
            and state.span_px and state.span_px > 0.0
        )
        tracker_center_px = state.center_px if tracker_visible else None
        tracker_speed = None
        prev_tracker_center = self._debug_prev_tracker.get("center")
        prev_tracker_time = self._debug_prev_tracker.get("time")
        if tracker_visible and tracker_center_px is not None and prev_tracker_center is not None and prev_tracker_time is not None:
            dt_tracker = float(state.timestamp) - float(prev_tracker_time)
            if dt_tracker > 1e-6:
                dx_px = float(tracker_center_px[0]) - float(prev_tracker_center[0])
                dy_px = float(tracker_center_px[1]) - float(prev_tracker_center[1])
                tracker_speed = math.hypot(dx_px, dy_px) / dt_tracker
        if tracker_visible and tracker_center_px is not None:
            self._debug_prev_tracker = {
                "center": (float(tracker_center_px[0]), float(tracker_center_px[1])),
                "time": float(state.timestamp),
            }
        else:
            self._debug_prev_tracker = {"center": None, "time": float(state.timestamp)}

        club_game_center = None
        if putter_visible and putter_center is not None:
            club_game_center = (float(putter_center[0]), float(putter_center[1]))
        elif tracker_visible and tracker_center_px is not None:
            club_game_center = self._tracker_px_to_game(tracker_center_px)

        ball_x = float(self.ball_pos[0])
        ball_y = float(self.ball_pos[1])

        club_to_ball_distance = None
        angle_abs = None
        angle_signed = None
        club_heading_deg = None
        ball_heading_deg = None
        heading_vec = None
        if putter_visible and putter_state is not None:
            heading_angle = float(putter_state.get("angle_deg", 0.0))
            angle_rad = math.radians(heading_angle)
            heading_vec = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=float)
            club_heading_deg = heading_angle
        elif tracker_visible and state.direction_px is not None:
            dir_game = self._tracker_dir_to_game(state.center_px, state.direction_px)
            dir_norm = np.linalg.norm(dir_game)
            if dir_norm > 1e-6:
                heading_vec = dir_game / dir_norm
                club_heading_deg = math.degrees(math.atan2(heading_vec[1], heading_vec[0]))

        if club_game_center is not None:
            dx_ball = ball_x - float(club_game_center[0])
            dy_ball = ball_y - float(club_game_center[1])
            club_to_ball_distance = math.hypot(dx_ball, dy_ball)
            if club_to_ball_distance > 1e-6:
                ball_heading_deg = math.degrees(math.atan2(dy_ball, dx_ball))
        if heading_vec is not None and club_game_center is not None and club_to_ball_distance and club_to_ball_distance > 1e-6:
            ball_vec = np.array(
                [ball_x - float(club_game_center[0]), ball_y - float(club_game_center[1])],
                dtype=float,
            )
            ball_unit = ball_vec / club_to_ball_distance
            dot = float(np.clip(np.dot(heading_vec, ball_unit), -1.0, 1.0))
            angle_abs = math.degrees(math.acos(dot))
            cross = float(heading_vec[0] * ball_unit[1] - heading_vec[1] * ball_unit[0])
            angle_signed = angle_abs if cross >= 0 else -angle_abs

        border_width = float(self.viz.border.get_width())
        border_height = float(self.viz.border.get_height())

        entry = {
            "timestamp_wall": time.time(),
            "timestamp_tracker": float(state.timestamp),
            "tracking_area": {
                "frame_width": frame_width,
                "frame_height": frame_height,
                "frame_pixels": frame_width * frame_height if frame_width and frame_height else None,
                "overlay_area_game": float(self._tracker_last_area),
            },
            "map_border": {"width": border_width, "height": border_height},
            "ball": {"x": ball_x, "y": ball_y},
            "putter": {
                "visible": putter_visible,
                "center_game": _tuple_to_list(putter_center if putter_visible else None),
                "length": float(putter_state.get("length", 0.0)) if putter_state else None,
                "thickness": float(putter_state.get("thickness", 0.0)) if putter_state else None,
                "angle_deg": float(putter_state.get("angle_deg", 0.0)) if putter_state else None,
                "speed_game_per_s": putter_speed,
            },
            "tracker": {
                "visible": tracker_visible,
                "center_px": _tuple_to_list(tracker_center_px),
                "direction_px": _tuple_to_list(state.direction_px if tracker_visible else None),
                "span_px": float(state.span_px or 0.0) if tracker_visible else None,
                "speed_px_per_s": tracker_speed,
            },
            "reference_px": _tuple_to_list(reference_px),
            "club_game": {
                "center": _tuple_to_list(club_game_center),
                "heading_deg": club_heading_deg,
                "distance_to_ball": club_to_ball_distance,
                "angle_to_ball_deg": angle_abs,
                "angle_to_ball_signed_deg": angle_signed,
                "ball_heading_from_club_deg": ball_heading_deg,
            },
        }
        self._write_debug_log(entry)

    def _compute_interference_profile(self, density):
        width_cfg = getattr(self.cfg, 'interference_probe_width', 4)
        width = int(max(1, width_cfg))
        if isinstance(density, np.ndarray):
            width = min(width, density.shape[1])
            slice_cpu = density[:, -width:]
            profile = slice_cpu.mean(axis=1, dtype=np.float32)
        else:
            xp = self.be.xp
            width = min(width, density.shape[1])
            slice_xp = density[:, -width:]
            profile_xp = xp.mean(slice_xp, axis=1)
            profile = self.be.to_cpu(profile_xp).astype(np.float32)
        np.clip(profile, 0.0, None, out=profile)
        max_val = float(profile.max())
        if max_val > 0.0:
            profile /= max_val
        return profile

    def _save_normal_state(self):
        if self._saved_state is None:
            self._saved_state = {
                'map_kind': self.cfg.map_kind,
                'mode': self.mode,
                'wave_profile': getattr(self.cfg, 'wave_initial_profile', 'packet'),
                'shot_stop_mode': getattr(self.cfg, 'shot_stop_mode', 'time'),
                'edge_boundary': getattr(self.cfg, 'edge_boundary', 'reflect'),
                'enable_mouse_swing': bool(getattr(self.cfg, 'enable_mouse_swing', False)),
                'boost_enabled': bool(getattr(self.cfg, 'boost_hole_probability', False)),
                'shot_time_limit': getattr(self.cfg, 'shot_time_limit', None),
                'movement_speed_scale': float(getattr(self.cfg, 'movement_speed_scale', 1.0)),
                'boost_increment': float(getattr(self.cfg, 'boost_hole_probability_increment', 0.0)),
                'sink_prob_threshold': float(getattr(self.cfg, 'sink_prob_threshold', 0.25)),
                'single_wall_thickness_factor': float(getattr(self.cfg, 'single_wall_thickness_factor', 1.0)),
            }

    def _restore_normal_state(self, keep_saved: bool = False, announce: bool = True):
        if self._saved_state is None:
            return False
        state = self._saved_state
        self._active_course = None
        self._course_stage = None

        self.cfg.edge_boundary = state['edge_boundary']
        self.course.edge_boundary = state['edge_boundary']
        self.cfg.wave_initial_profile = state['wave_profile']
        self.cfg.shot_stop_mode = state['shot_stop_mode']
        self.cfg.enable_mouse_swing = state['enable_mouse_swing']
        self.cfg.boost_hole_probability = state.get('boost_enabled', getattr(self.cfg, 'boost_hole_probability', False))
        self.cfg.boost_hole_probability_increment = state.get('boost_increment', self.cfg.boost_hole_probability_increment)
        self.cfg.movement_speed_scale = state.get('movement_speed_scale', self.cfg.movement_speed_scale)
        self.cfg.shot_time_limit = state.get('shot_time_limit', self.cfg.shot_time_limit)
        self.cfg.sink_prob_threshold = state.get('sink_prob_threshold', self.cfg.sink_prob_threshold)
        self.cfg.single_wall_thickness_factor = state.get('single_wall_thickness_factor', getattr(self.cfg, 'single_wall_thickness_factor', 1.0))
        self._apply_movement_speed_tuning()
        self.cfg.map_kind = state['map_kind']
        self.course.set_map(self.cfg.map_kind)
        self._refresh_course_assets()
        self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
        self.viz.set_course_patches(self.course.course_patches)
        self._set_mode(state['mode'])
        if self._config_panel_active:
            self._sync_config_panel_values()
        if not keep_saved:
            self._saved_state = None
        self._reset(ball_only=False)
        if announce:
            print("[course] Returned to standard mode.")
        return True

    def _cycle_course(self, course_name: str):
        presets = self._course_presets.get(course_name)
        if not presets:
            return
        if self.shot_in_progress:
            print(f"[{course_name}] Finish the current shot before changing stages.")
            return

        trigger_key = self._course_trigger_keys.get(course_name, '?')

        if self._active_course != course_name:
            if self._active_course is not None:
                self._restore_normal_state(keep_saved=True, announce=False)
            self._save_normal_state()
            self._active_course = course_name
            stage_idx = 0
        else:
            current = self._course_stage if self._course_stage is not None else -1
            stage_idx = (current + 1) % len(presets)

        self._apply_course_stage(course_name, stage_idx, trigger_key)

    def _apply_course_stage(self, course_name: str, stage_idx: int, trigger_key: str):
        presets = self._course_presets.get(course_name)
        if not presets:
            return
        preset = presets[stage_idx]

        if 'edge_boundary' in preset:
            edge_mode = preset['edge_boundary']
            self.cfg.edge_boundary = edge_mode
            self.course.edge_boundary = edge_mode

        if 'shot_stop_mode' in preset:
            self.cfg.shot_stop_mode = preset['shot_stop_mode']

        if 'wave_profile' in preset:
            self.cfg.wave_initial_profile = preset['wave_profile']

        if 'single_wall_thickness_factor' in preset:
            self.cfg.single_wall_thickness_factor = float(np.clip(preset['single_wall_thickness_factor'], 0.05, 2.5))
            if self._config_panel_active and 'wall' in self._panel_sliders:
                self._panel_updating = True
                try:
                    self._panel_sliders['wall'].set_val(self.cfg.single_wall_thickness_factor)
                finally:
                    self._panel_updating = False

        if 'map' in preset:
            self.cfg.map_kind = preset['map']
            self.course.set_map(self.cfg.map_kind)
            self._refresh_course_assets()
            self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
            self.viz.set_course_patches(self.course.course_patches)

        if 'mode' in preset:
            self._set_mode(preset['mode'])

        self._active_course = course_name
        self._course_stage = stage_idx
        self._apply_movement_speed_tuning()

        if self._config_panel_active:
            self._sync_config_panel_values()

        self._reset(ball_only=False)

        desc = preset.get('description', '')
        total = len(presets)
        print(f"[{course_name}] Stage {stage_idx + 1}/{total} - {desc} (press '{trigger_key}' to advance, 'r' to return)")

    def _toggle_edge_boundary(self):
        if self.shot_in_progress:
            print("[b] cannot toggle edge boundary during a shot")
            return
        old = getattr(self.cfg, 'edge_boundary', 'reflect')
        new = 'absorb' if old == 'reflect' else 'reflect'
        self.cfg.edge_boundary = new
        self.course.edge_boundary = new
        self.course.set_map(self.cfg.map_kind)
        self._refresh_course_assets()
        self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
        self.viz.set_course_patches(self.course.course_patches)
        self._reset(ball_only=False)
        self._announce_switch('b', 'edge_boundary', old, new)

    def _toggle_wave_profile(self):
        if self.shot_in_progress:
            print("[w] cannot toggle wave profile during a shot")
            return
        old = getattr(self.cfg, 'wave_initial_profile', 'packet').lower()
        new = 'front' if old != 'front' else 'packet'
        self.cfg.wave_initial_profile = new
        self._wavefront_profile = new
        self._wavefront_active = (new == 'front')
        if self._wavefront_active:
            self._wavefront_dir = np.array([1.0, 0.0], dtype=np.float32)
            self._wavefront_kmag = 0.0
        if self._mode_allows_quantum():
            self._draw_idle_wave_preview()
        self._announce_switch('w', 'wave_initial_profile', old, new)

    def _toggle_shot_stop_mode(self):
        old = getattr(self.cfg, 'shot_stop_mode', 'time').lower()
        new = 'friction' if old != 'friction' else 'time'
        self.cfg.shot_stop_mode = new
        self._announce_switch('t', 'shot_stop_mode', old, new)
        if self._config_panel_active:
            self._update_panel_stats()
            self._panel_draw_idle()

    def _toggle_mouse_swing(self):
        old = bool(getattr(self.cfg, 'enable_mouse_swing', False))
        new = not old
        self.cfg.enable_mouse_swing = new
        if new:
            self._set_tracker_force_disabled(True)
        else:
            self._set_tracker_force_disabled(False)
        if not new:
            self.cursor_inside = False
            self.indicator_pos = None
            self.last_mouse_pos = None
            self.last_mouse_t = None
            self.viz.indicator_patch.set_visible(False)
            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()
        self._announce_switch('g', 'enable_mouse_swing', old, new)

    def _toggle_tracker_overlay(self):
        if self._tracker_decoupled:
            print('[o] Tracker overlay is decoupled from Matplotlib; restart with --no-decouple-tracker-overlay to enable in-scene drawing.')
            return
        old_state = 'visible' if self._tracker_overlay_enabled else 'hidden'
        self._tracker_overlay_enabled = not self._tracker_overlay_enabled
        if not self._tracker_overlay_enabled:
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)
        new_state = 'visible' if self._tracker_overlay_enabled else 'hidden'
        self._announce_switch('o', 'tracker overlay', old_state, new_state)

    def _set_tracker_force_disabled(self, disabled: bool):
        disabled = bool(disabled)
        if self._tracker_force_disabled == disabled:
            return
        self._tracker_force_disabled = disabled
        if disabled:
            self._tracker_area_valid = False
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)
            if self.tracker:
                self.tracker.pop_hits()
        else:
            # Allow tracker poll to re-enable overlay when geometry is valid
            if self.tracker:
                self._update_tracker_reference()
        self._refresh_tracker_overlay_mode()

    def _print_hotkey_help(self):
        BLUE = "\033[34m\033[4m"
        RESET = "\033[0m"
        def fmt_state(value):
            return f"{BLUE}{value}{RESET}"

        course_demo_state = 'inactive'
        course_show_state = 'inactive'
        if self._active_course == 'quantum_demo' and self._course_stage is not None:
            course_demo_state = f"stage {self._course_stage + 1}/{len(self._course_presets['quantum_demo'])}"
        elif self._active_course == 'advanced_showcase' and self._course_stage is not None:
            course_show_state = f"stage {self._course_stage + 1}/{len(self._course_presets['advanced_showcase'])}"
        elif self._active_course == 'quantum_demo':
            course_demo_state = "stage 1"
        elif self._active_course == 'advanced_showcase':
            course_show_state = "stage 1"

        measurement_state = 'enabled' if self._mode_allows_measure() else 'disabled'
        info_state = 'on' if self.show_info else 'off'
        mouse_state = 'on' if getattr(self.cfg, 'enable_mouse_swing', False) else 'off'
        panel_state = 'open' if self._config_panel_active else 'closed'
        interference_state = 'visible' if self.show_interference else 'hidden'
        tracker_overlay_state = 'visible' if self._tracker_overlay_enabled else 'hidden'
        if self._playback_path:
            try:
                playback_state = Path(self._playback_path).name
            except Exception:
                playback_state = 'ready'
        else:
            playback_state = 'idle'

        entries = [
            ("q", "Quit the game", None),
            ("r", "Reset ball and wave", None),
            ("tab", "Cycle obstacle map", self.cfg.map_kind),
            ("c", "Cycle display mode", self.mode),
            ("m", "Force quantum measurement", measurement_state),
            ("i", "Info overlay / post-shot measure", info_state),
            ("#", "Cycle quantum demo course", course_demo_state),
            ("-", "Cycle advanced showcase course", course_show_state),
            ("b", "Toggle edge boundary reflect/absorb", self.cfg.edge_boundary),
            ("w", "Toggle wave initial profile packet/front", getattr(self.cfg, 'wave_initial_profile', 'packet')),
            ("t", "Toggle shot stop mode time/friction", getattr(self.cfg, 'shot_stop_mode', 'time')),
            ("g", "Toggle mouse swing control", mouse_state),
            ("o", "Toggle tracker overlay visibility", tracker_overlay_state),
            ("u", "Toggle control panel window", panel_state),
            ("p", "Cycle background image", self._background_state_label()),
            ("l", "Toggle interference profile view", interference_state),
            ("d", "Play latest recording (if available)", playback_state),
            ("h", "Show this hotkey list", None),
        ]
        lines = []
        for key, desc, state in entries:
            if state is not None:
                lines.append(f"  {key:<3} - {desc} (current: {fmt_state(state)})")
            else:
                lines.append(f"  {key:<3} - {desc}")
        print("\nHotkeys:\n" + "\n".join(lines) + "\n")

    def _build_background_cycle(self) -> list[Path | None]:
        choices: list[Path | None] = [None]
        seen: set[Path] = set()
        search_dirs: list[Path] = []
        custom_dir = getattr(self.cfg, 'background_cycle_dir', None)
        if custom_dir:
            search_dirs.append(Path(str(custom_dir)).expanduser())
        search_dirs.append(Path.cwd() / "BackgroundImages")
        search_dirs.append(Path(__file__).resolve().parent.parent / "BackgroundImages")
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        for directory in search_dirs:
            try:
                resolved_dir = directory.resolve()
            except Exception:
                resolved_dir = directory
            if not resolved_dir.is_dir():
                continue
            for child in sorted(resolved_dir.iterdir()):
                if not child.is_file():
                    continue
                if child.suffix.lower() not in suffixes:
                    continue
                try:
                    resolved = child.resolve()
                except Exception:
                    resolved = child
                if resolved in seen:
                    continue
                seen.add(resolved)
                choices.append(resolved)
        cfg_path = getattr(self.cfg, 'background_image_path', None)
        if cfg_path:
            try:
                resolved_cfg = Path(str(cfg_path)).expanduser().resolve()
            except Exception:
                resolved_cfg = Path(str(cfg_path)).expanduser()
            if resolved_cfg.is_file() and resolved_cfg not in seen:
                choices.append(resolved_cfg)
        return choices

    def _find_background_index(self, target: Path | None) -> int | None:
        if target is None:
            return 0 if getattr(self, '_background_cycle', None) else None
        try:
            target_resolved = Path(str(target)).resolve()
        except Exception:
            target_resolved = Path(str(target))
        for idx, entry in enumerate(self._background_cycle):
            if entry is None:
                continue
            if self._paths_equal(entry, target_resolved):
                return idx
        return None

    @staticmethod
    def _paths_equal(a: Path, b: Path) -> bool:
        try:
            return a.resolve() == b.resolve()
        except Exception:
            return a == b

    def _cycle_background(self) -> None:
        if not self._background_cycle:
            self._background_cycle = [None]
        attempts = 0
        total = len(self._background_cycle) or 1
        while attempts < total:
            self._background_index = (self._background_index + 1) % len(self._background_cycle)
            target = self._background_cycle[self._background_index]
            ok = self.viz.set_background_image(target)
            if ok:
                if target is None:
                    self.cfg.background_image_path = None
                    print("[bg] Background set to black.")
                else:
                    try:
                        resolved = target.resolve()
                    except Exception:
                        resolved = target
                    self.cfg.background_image_path = str(resolved)
                    print(f"[bg] Background set to '{resolved.name}'.")
                return
            problematic = self._background_cycle.pop(self._background_index)
            if not self._background_cycle:
                self._background_cycle = [None]
                self._background_index = 0
                self.viz.set_background_image(None)
                self.cfg.background_image_path = None
                print("[bg] Reverting to black background (no valid images).")
                return
            if self._background_index >= len(self._background_cycle):
                self._background_index = 0
            if problematic is not None:
                try:
                    name = problematic.name
                except Exception:
                    name = str(problematic)
                print(f"[bg] Skipping invalid background '{name}'.")
            total = len(self._background_cycle)
            attempts += 1
        print("[bg] Unable to change background (no valid images).")

    def _background_state_label(self) -> str:
        path = self.viz.current_background_path()
        if path is None:
            return "black"
        try:
            return Path(path).name
        except Exception:
            return str(path)

    def _announce_switch(self, key: str, label: str, old, new):
        BLUE = "\033[34m\033[4m"
        GREEN = "\033[32m\033[4m"
        RESET = "\033[0m"
        print(f"[{key}] {label}: {BLUE}{old}{RESET} \u2192 {GREEN}{new}{RESET}")

    # ----- playback helpers
    def _play_recording(self, path: str | Path = None):
        if path is None:
            path = Path(os.getcwd()) / "QuantumMinigolfDemo.mp4"

        if self.shot_in_progress:
            print('[d] Finish the current shot before playing a recording.')
            return
        if self._playback_timer is not None or self._playback_image is not None:
            self._stop_playback()
            return

        target = Path(path)
        if not target.exists():
            print(f"[d] Recording not found: {target}")
            return
        try:
            import imageio.v3 as iio  # type: ignore
        except Exception:
            print('[d] The imageio package is required to play back recordings.')
            return

        try:
            meta = iio.immeta(target, plugin='ffmpeg')
            fps = float(meta.get('fps', 30.0) or 30.0)
        except Exception:
            fps = 30.0
        fps = max(1.0, fps)
        playback_speed = float(max(0.1, getattr(self.cfg, 'video_playback_speed', self._video_playback_speed)))
        self._video_playback_speed = playback_speed

        try:
            frame_iter = iio.imiter(target, plugin='ffmpeg')
        except Exception as exc:
            print(f"[d] Unable to open recording: {exc}")
            return

        try:
            first_frame = next(frame_iter)
        except StopIteration:
            print('[d] Recording is empty.')
            return
        except Exception as exc:
            print(f"[d] Failed reading recording: {exc}")
            return

        self._playback_iter = frame_iter
        self._playback_close = getattr(frame_iter, 'close', None)
        self._playback_path = target
        self.viz.ax.set_visible(False)
        if self.show_interference:
            self.viz.set_interference_visible(False)
        self._playback_image = self.viz.fig.figimage(first_frame, origin='upper', zorder=50)
        interval_ms = max(1, int(round(1000.0 / (fps * playback_speed))))
        timer = self.viz.fig.canvas.new_timer(interval=interval_ms)
        timer.add_callback(self._advance_playback_frame)
        self._playback_timer = timer
        self._playback_hold = False
        timer.start()
        self.viz.fig.canvas.draw_idle()
        print('[d] Playing recording. Press d again to stop playback.')

    def _advance_playback_frame(self):
        if self._playback_iter is None or self._playback_image is None:
            self._stop_playback()
            return
        try:
            frame = next(self._playback_iter)
        except StopIteration:
            self._finalize_playback_hold()
            return
        except Exception as exc:
            print(f"[d] Playback stopped: {exc}")
            self._stop_playback()
            return
        self._playback_image.set_data(frame)
        self.viz.fig.canvas.draw_idle()

    def _finalize_playback_hold(self):
        if self._playback_timer is not None:
            try:
                self._playback_timer.stop()
            except Exception:
                pass
            self._playback_timer = None
        if self._playback_iter is not None:
            close_fn = getattr(self._playback_iter, 'close', None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if callable(self._playback_close):
            try:
                self._playback_close()
            except Exception:
                pass
        self._playback_iter = None
        self._playback_close = None
        self._playback_hold = True
        print('[d] Playback finished. Press d to close the preview.')

    def _stop_playback(self):
        if self._playback_timer is not None:
            try:
                self._playback_timer.stop()
            except Exception:
                pass
            self._playback_timer = None
        if self._playback_iter is not None:
            close_fn = getattr(self._playback_iter, 'close', None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
        if callable(self._playback_close):
            try:
                self._playback_close()
            except Exception:
                pass
        if self._playback_image is not None:
            try:
                self._playback_image.remove()
            except Exception:
                pass
            self._playback_image = None
        self._playback_close = None
        self._playback_iter = None
        self._playback_path = None
        self._playback_hold = False
        self.viz.ax.set_visible(True)
        if self.show_interference and self._interference_profile is not None:
            self.viz.set_interference_visible(True)
            self.viz.update_interference_pattern(self._interference_profile)
        else:
            self.viz.fig.canvas.draw_idle()

    def _toggle_config_panel(self):
        if Slider is None:
            print('[control panel] Matplotlib Slider widgets unavailable; control panel disabled.')
            return
        if self._config_panel_active:
            self._deactivate_config_panel()
        else:
            self._activate_config_panel()

    def _on_panel_key(self, event):
        if event is None:
            return
        self._on_key(event)

    def _activate_config_panel(self):
        if self._config_panel_active or Slider is None:
            if Slider is None:
                print('[control panel] Matplotlib Slider widgets unavailable; control panel disabled.')
            return

        fig = plt.figure(figsize=(11.3, 6.0))
        fig.patch.set_facecolor('black')
        try:
            fig.canvas.manager.set_window_title('Quantum Mini-Golf - Control Panel')
        except Exception:
            pass

        self._panel_fig = fig
        self._panel_close_cid = fig.canvas.mpl_connect('close_event', self._on_panel_close)
        self._panel_key_cid = fig.canvas.mpl_connect('key_press_event', self._on_panel_key)

        self._panel_axes_list.clear()
        self._panel_sliders.clear()
        self._panel_elements.clear()

        bg_ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
        bg_ax.set_facecolor((0.05, 0.05, 0.05, 0.95))
        bg_ax.set_xticks([])
        bg_ax.set_yticks([])
        for spine in bg_ax.spines.values():
            spine.set_visible(False)
        self._panel_axes_list.append(bg_ax)
        bg_ax.text(0.5, 0.97, 'Control Panel', color='white', fontsize=12, fontweight='bold', ha='center', va='top', transform=bg_ax.transAxes)

        slider_left = 0.18
        slider_width = 0.64
        slider_height = 0.042
        slider_gap = 0.055
        slider_y = 0.82

        slider_specs = [
            ('move', 'Move Speed',     2.0, 25.0, float(self.cfg.movement_speed_scale),    0.1, '#9467bd', self._on_movement_speed_change),
            ('time', 'Shot Time',      10.0, 200.0, float(self.cfg.shot_time_limit or 75.0),    5.0, '#8c564b', self._on_shot_time_limit_change),
            ('wall', 'Wall Thickness', 0.05,2.5,  float(getattr(self.cfg, 'single_wall_thickness_factor', 1.0)), 0.05, '#17becf', self._on_wall_thickness_change),
            ('tracker_thresh', 'LED Recognition Threshold', 1.0, 250.0, float(getattr(self.cfg, 'tracker_threshold', 55)), 1.0, '#bcbd22', self._on_tracker_threshold_change),
            ('tracker_speed',  'Putter Speed Increase', 0.25, 4.0, float(self.cfg.tracker_speed_scale / max(self._tracker_speed_base, 1e-6)), 0.05, '#e377c2', self._on_tracker_speed_scale_change),
            ('tracker_size',  'Putter Size',      0.1, 1.5, float(max(0.1, getattr(self.cfg, 'tracker_length_scale', 0.3))), 0.05, '#1f77b4', self._on_tracker_size_scale_change),
        ]

        for name, label, vmin, vmax, val, step, color, callback in slider_specs:
            ax = fig.add_axes([slider_left, slider_y, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=val, valstep=step)
            self._format_slider(slider, color=color)
            slider.on_changed(callback)
            self._panel_sliders[name] = slider
            self._panel_axes_list.append(ax)
            slider_y -= slider_gap

        self._panel_elements['time_slider'] = self._panel_sliders['time']

        self._panel_updating = False
        self._config_panel_active = True
        self.cfg.show_control_panel = True
        self._sync_config_panel_values()
        self._panel_draw_idle()
        try:
            if hasattr(self._panel_fig, 'show'):
                self._panel_fig.show()
        except Exception:
            pass

    def _deactivate_config_panel(self):
        if not self._config_panel_active:
            return
        self._config_panel_active = False
        self._panel_updating = False
        self.cfg.show_control_panel = False
        if self._panel_fig is not None:
            if self._panel_key_cid is not None:
                try:
                    self._panel_fig.canvas.mpl_disconnect(self._panel_key_cid)
                except Exception:
                    pass
                self._panel_key_cid = None
            try:
                plt.close(self._panel_fig)
            except Exception:
                self._on_panel_close(None)
        else:
            self._panel_axes_list.clear()
            self._panel_sliders.clear()
            self._panel_elements.clear()
            self._panel_close_cid = None
            self._panel_key_cid = None

    def _on_panel_close(self, _event):
        if self._panel_key_cid is not None:
            canvas = getattr(_event, 'canvas', None)
            if canvas is not None:
                try:
                    canvas.mpl_disconnect(self._panel_key_cid)
                except Exception:
                    pass
        self._panel_axes_list.clear()
        self._panel_sliders.clear()
        self._panel_elements.clear()
        self._panel_updating = False
        self._panel_fig = None
        self._panel_close_cid = None
        self._panel_key_cid = None
        self._config_panel_active = False
        self.cfg.show_control_panel = False

    def _panel_draw_idle(self):
        if self._panel_fig is not None:
            try:
                self._panel_fig.canvas.draw_idle()
            except Exception:
                pass

    def _update_panel_stats(self):
        stats = self._panel_elements.get('stats_text')
        if stats is None:
            return
        limit = 'inf' if self.cfg.shot_time_limit is None else f"{self.cfg.shot_time_limit:.0f}s"
        stop_mode = str(getattr(self.cfg, 'shot_stop_mode', 'time'))
        move_slider = float(getattr(self.cfg, 'movement_speed_scale', 17.5))
        move_factor = getattr(self, '_movement_speed_factor', move_slider)
        stats.set_text(
            "Target FPS: {fps:.0f}\nShot limit: {limit}\nStop mode: {stop_mode}\nMove slider: {slider:.2f}\nSpeed factor: {factor:.2f}x".format(
                fps=self.cfg.target_fps,
                limit=limit,
                stop_mode=stop_mode,
                slider=move_slider,
                factor=move_factor,
            )
        )

    def _sync_config_panel_values(self):
        if not self._config_panel_active or not self._panel_sliders:
            return
        ratio = float(self.cfg.tracker_speed_scale / max(self._tracker_speed_base, 1e-6))
        values = {
            'lin': float(self.cfg.shot_friction_linear),
            'quad': float(self.cfg.shot_friction_quadratic),
            'min': float(self.cfg.shot_friction_min_scale),
            'sink': float(self.cfg.sink_prob_threshold),
            'move': float(np.clip(self.cfg.movement_speed_scale, self._movement_slider_bounds[0], self._movement_slider_bounds[1])),
            'time': float(np.clip(self.cfg.shot_time_limit if self.cfg.shot_time_limit is not None else 75.0, 10.0, 200.0)),
            'wall': float(np.clip(getattr(self.cfg, 'single_wall_thickness_factor', 1.0), 0.05, 2.5)),
            'tracker_thresh': float(np.clip(getattr(self.cfg, 'tracker_threshold', 55), 1.0, 250.0)),
            'tracker_speed': float(np.clip(ratio, 0.25, 4.0)),
            'tracker_size': float(np.clip(getattr(self.cfg, 'tracker_length_scale', 0.3), 0.1, 1.5)),
        }
        self._panel_updating = True
        try:
            for key, value in values.items():
                slider = self._panel_sliders.get(key)
                if slider is not None:
                    slider.set_val(value)
        finally:
            self._panel_updating = False
        self._refresh_slider_texts()

    def _format_slider(self, slider: Slider, color: str):
        slider.ax.set_facecolor('#222222')
        slider.poly.set_color(color)
        slider.label.set_color('white')
        slider.valtext.set_color('white')

    def _refresh_slider_texts(self, draw=True):
        if not self._config_panel_active:
            return
        sliders = self._panel_sliders
        if 'lin' in sliders:
            sliders['lin'].valtext.set_text(f"{self.cfg.shot_friction_linear:.3f}")
        if 'quad' in sliders:
            sliders['quad'].valtext.set_text(f"{self.cfg.shot_friction_quadratic:.3f}")
        if 'min' in sliders:
            sliders['min'].valtext.set_text(f"{self.cfg.shot_friction_min_scale:.3f}")
        if 'sink' in sliders:
            sliders['sink'].valtext.set_text(f"{self.cfg.sink_prob_threshold:.2f}")
        if 'move' in sliders:
            sliders['move'].valtext.set_text(f"{self._movement_speed_factor:.2f}x")
        if 'time' in sliders:
            if self.cfg.shot_time_limit is None:
                sliders['time'].valtext.set_text('inf')
            else:
                sliders['time'].valtext.set_text(f"{self.cfg.shot_time_limit:.0f}")
        if 'wall' in sliders:
            sliders['wall'].valtext.set_text(f"{getattr(self.cfg, 'single_wall_thickness_factor', 1.0):.2f}")
        if 'tracker_thresh' in sliders:
            sliders['tracker_thresh'].valtext.set_text(f"{int(round(getattr(self.cfg, 'tracker_threshold', 55)))}")
        if 'tracker_speed' in sliders:
            factor = float(self.cfg.tracker_speed_scale / max(self._tracker_speed_base, 1e-6))
            sliders['tracker_speed'].valtext.set_text(f"{factor:.2f}x")
        if 'tracker_size' in sliders:
            sliders['tracker_size'].valtext.set_text(f"{float(max(0.1, self.cfg.tracker_length_scale)):.2f}x")
        if 'cubic_text' in self._panel_elements:
            cubic = max(0.0, 1.0 - self.cfg.shot_friction_linear - self.cfg.shot_friction_quadratic)
            self._panel_elements['cubic_text'].set_text(f"Friction Cubic: {cubic:.3f}")
        self._update_panel_stats()
        if draw:
            self._panel_draw_idle()

    def _on_friction_slider_change(self, _=None):
        if not self._config_panel_active or 'lin' not in self._panel_sliders:
            return
        if getattr(self, '_panel_updating', False):
            return
        lin = float(self._panel_sliders['lin'].val)
        quad = float(self._panel_sliders['quad'].val)
        if lin + quad > 0.98:
            excess = (lin + quad) - 0.98
            self._panel_updating = True
            try:
                if lin >= quad:
                    lin = max(0.0, lin - excess / 2.0)
                    quad = max(0.0, quad - excess / 2.0)
                else:
                    quad = max(0.0, quad - excess)
                self._panel_sliders['lin'].set_val(lin)
                self._panel_sliders['quad'].set_val(quad)
            finally:
                self._panel_updating = False
            lin = float(self._panel_sliders['lin'].val)
            quad = float(self._panel_sliders['quad'].val)

        cubic = max(0.0, 1.0 - lin - quad)
        self.cfg.shot_friction_linear = lin
        self.cfg.shot_friction_quadratic = quad
        self.cfg.shot_friction_cubic = cubic
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_min_scale_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        self.cfg.shot_friction_min_scale = max(0.0, min(0.5, float(val)))
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_sink_threshold_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        self.cfg.sink_prob_threshold = max(0.0, min(1.0, float(val)))
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_movement_speed_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        self.cfg.movement_speed_scale = float(val)
        self._apply_movement_speed_tuning()
        self._panel_draw_idle()

    def _on_shot_time_limit_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        value = float(np.clip(val, 10.0, 200.0))
        self.cfg.shot_time_limit = value
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_wall_thickness_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        thickness = float(np.clip(val, 0.05, 2.5))
        self.cfg.single_wall_thickness_factor = thickness
        if self.cfg.map_kind in {'single_wall', 'single_slit', 'double_slit', 'new_map'}:
            self.course.set_map(self.cfg.map_kind)
            self._refresh_course_assets()
            self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
            self.viz.set_course_patches(self.course.course_patches)
            if self._mode_allows_quantum():
                self._draw_idle_wave_preview()
            elif self._mode_allows_classical():
                self.viz.draw_frame(np.zeros((self.Ny, self.Nx), dtype=np.float32), plot_wave=False)
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_tracker_threshold_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        thresh = int(round(float(val)))
        thresh = max(1, min(250, thresh))
        self.cfg.tracker_threshold = thresh
        if self.tracker_cfg is not None:
            self.tracker_cfg.threshold = thresh
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_tracker_speed_scale_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        factor = float(np.clip(val, 0.25, 4.0))
        self.cfg.tracker_speed_scale = self._tracker_speed_base * factor
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_tracker_size_scale_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        factor = float(np.clip(val, 0.1, 1.5))
        self.cfg.tracker_length_scale = factor
        self.cfg.tracker_thickness_scale = factor
        self._apply_tracker_size_scale()
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _reset(self, ball_only=False):
        start_x = int(
            round(max(8, min(self.Nx - 8, self.Nx * self.cfg.ball_start_x_frac))))
        self.ball_pos = np.array(
            [float(start_x), self.Ny / 2], dtype=np.float32)
        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])
        self._reset_shot_counter()

        plot_wave = self._mode_allows_quantum()
        self.viz.set_wave_visible(plot_wave)
        self.viz.clear_classical_overlay()
        self.viz.clear_wave_overlay(plot_wave)
        self.show_interference = False
        self._interference_profile = None
        self.viz.set_interference_visible(False)
        self._stop_playback()

        self._wavefront_profile = str(getattr(self.cfg, 'wave_initial_profile', 'packet')).lower()
        self._wavefront_active = self._wavefront_profile == 'front'
        if self._wavefront_active:
            self._wavefront_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            self._wavefront_dir = np.array([1.0, 0.0], dtype=np.float32)
        self._wavefront_kmag = 0.0

        if plot_wave:
            self._draw_idle_wave_preview()
        else:
            self.viz.draw_frame(
                np.zeros((self.Ny, self.Nx), dtype=np.float32), plot_wave=False)
        self.viz.set_ball_visible(self._mode_allows_classical())
        self.viz.measure_marker.set_visible(False)
        self.viz.measure_point.set_visible(False)
        self._last_density_cpu = None
        self._last_measure_xy = None
        self._last_measure_prob = None
        self._last_ex = None
        self._last_ey = None
        self._last_phase_cpu = None
        self.viz.set_info_visibility(False)
        self.viz.set_wave_path_label(False)
        self.viz.show_messages(False, False)
        self.viz.update_title(self._title_text())

        self.game_over = False
        self.shot_in_progress = False

        self.cursor_inside = False
        self.indicator_pos = None
        self.last_mouse_pos = None
        self.last_mouse_t = None
        self.viz.indicator_patch.set_visible(False)

        # reset bias factor
        self._boost_factor0 = float(self.cfg.boost_hole_probability_factor)
        if self.tracker:
            self._update_tracker_reference()

    def _double_reset(self):
        self._reset()
        self._reset()
        self._abort_shot_requested = False

    # ----- swing & shot
    def _compute_kvec_from_swing(self, v_vec, speed_scalar):
        s = float(max(speed_scalar, 0.0))
        if s < self.cfg.impact_min_speed:
            return None
        norm_v = float(np.linalg.norm(v_vec))
        if norm_v < 1e-9:
            return None

        kmax = float(self.be.to_cpu(self.k_max))
        # Dimensionless speed above threshold
        s_rel = max(0.0, s - self.cfg.impact_min_speed) / \
            max(1e-6, self.cfg.impact_min_speed)

        # Map speed to a fraction in [kmin_frac, kmax_frac] using a smooth tanh curve
        fmin = float(self.cfg.kmin_frac)
        fmax = float(self.cfg.kmax_frac)
        frac_dynamic = fmin + (fmax - fmin) * \
            np.tanh(self.cfg.swing_power_scale * s_rel)
        frac_neutral = 0.5 * (fmin + fmax)
        weight = float(np.clip(getattr(self.cfg, 'tunneling_speed_weight', 1.0), 0.0, 1.0))
        frac = frac_neutral + weight * (frac_dynamic - frac_neutral)
        frac = float(np.clip(frac, fmin, fmax))

        kmag = frac * kmax
        direction = v_vec / norm_v
        kvec = direction * kmag
        return kvec.astype(np.float32)

    @staticmethod
    def _segment_ball_intersection(p0, p1, center, radius):
        if p0 is None or p1 is None:
            return None
        p0 = np.asarray(p0, dtype=np.float32)
        p1 = np.asarray(p1, dtype=np.float32)
        center = np.asarray(center, dtype=np.float32)
        radius = float(radius)
        d = p1 - p0
        a = float(np.dot(d, d))
        if a <= 1e-12:
            # Degenerate segment -> just check endpoint
            dist = float(np.linalg.norm(p0 - center))
            if dist <= radius + 1e-6:
                return 0.0, p0
            return None
        f = p0 - center
        b = 2.0 * float(np.dot(f, d))
        c = float(np.dot(f, f) - radius * radius)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(max(disc, 0.0))
        inv_denom = 1.0 / (2.0 * a)
        t_candidates = [(-b - sqrt_disc) * inv_denom, (-b + sqrt_disc) * inv_denom]
        t_hit = None
        for t in sorted(t_candidates):
            if -1e-6 <= t <= 1.0 + 1e-6:
                t_hit = float(np.clip(t, 0.0, 1.0))
                break
        if t_hit is None:
            return None
        hit_point = p0 + t_hit * d
        return t_hit, hit_point

    def _check_collision_and_shoot(self, indicator_pos, v_vec, prev_pos=None, dt=None):
        if self.shot_in_progress or self.game_over:
            return
        indicator_pos = np.asarray(indicator_pos, dtype=np.float32)
        contact_r = float(self.cfg.ball_r + self.cfg.indicator_r)
        d = float(np.linalg.norm(indicator_pos - self.ball_pos))
        segment_hit = None
        if d > contact_r and prev_pos is not None:
            intersect = self._segment_ball_intersection(prev_pos, indicator_pos, self.ball_pos, contact_r)
            if intersect is not None:
                _, hit_point = intersect
                indicator_pos = hit_point.astype(np.float32)
                d = float(np.linalg.norm(indicator_pos - self.ball_pos))
                segment_hit = intersect
        if d > contact_r:
            return

        closing_speed = 0.0
        if prev_pos is not None and dt is not None and dt > 0.0:
            prev_dist = float(np.linalg.norm(prev_pos - self.ball_pos))
            curr_dist = float(np.linalg.norm(indicator_pos - self.ball_pos))
            closing = (prev_dist - curr_dist) / max(dt, 1e-6)
            if closing > 0.0:
                closing_speed = closing

        if closing_speed <= 0.0:
            sep_vec = indicator_pos - self.ball_pos
            sep_norm = float(np.linalg.norm(sep_vec))
            if sep_norm > 1e-6:
                closing_speed = max(
                    0.0, -float(np.dot(v_vec, sep_vec / sep_norm)))

        if segment_hit is not None:
            seg_len = float(np.linalg.norm(indicator_pos - np.asarray(prev_pos, dtype=np.float32)))
            if dt is not None and dt > 1e-6:
                closing_speed = max(closing_speed, seg_len / dt)
            elif v_vec is not None:
                closing_speed = max(closing_speed, float(np.linalg.norm(v_vec)))

        if closing_speed <= 0.0:
            return

        impact_vec = self.ball_pos - indicator_pos
        impact_norm = float(np.linalg.norm(impact_vec))
        normal_hat = impact_vec / impact_norm if impact_norm > 1e-6 else None

        contact_velocity = None
        if v_vec is not None:
            contact_velocity = np.asarray(v_vec, dtype=np.float32)
        if segment_hit is not None and prev_pos is not None and dt is not None and dt > 1e-6:
            prev_pos_arr = np.asarray(prev_pos, dtype=np.float32)
            contact_velocity = (indicator_pos - prev_pos_arr) / dt

        direction_vec: np.ndarray | None
        if contact_velocity is not None and normal_hat is not None:
            # Blend the club's tangential motion with the outward normal so the launch angle
            # mirrors the slider trajectory while still pushing the ball away from contact.
            forward = float(np.dot(contact_velocity, normal_hat))
            tangent = contact_velocity - forward * normal_hat
            normal_component = normal_hat * max(closing_speed, 1e-3)
            direction_vec = tangent + normal_component
            if np.linalg.norm(direction_vec) < 1e-6:
                direction_vec = normal_component
        elif contact_velocity is not None:
            direction_vec = contact_velocity
        elif normal_hat is not None:
            direction_vec = impact_vec.astype(np.float32)
        else:
            direction_vec = None

        if direction_vec is None or not np.isfinite(direction_vec).all():
            return

        direction_vec = np.asarray(direction_vec, dtype=np.float32)

        kvec = self._compute_kvec_from_swing(direction_vec, closing_speed)
        if kvec is None:
            return
        self.viz.indicator_patch.set_visible(False)
        self.viz.update_title(self._title_text())
        self._shoot(kvec)

    def _advance_classical_segment(self, pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
        """Advance the classical ball while reflecting exactly at contact."""
        if dt <= 0.0:
            return pos

        remaining = float(dt)
        nx_max = float(self.Nx)
        ny_max = float(self.Ny)
        ball_r = float(getattr(self.cfg, 'ball_r', 0.0))
        xmin = max(0.0, min(nx_max, ball_r))
        xmax = min(nx_max, max(xmin, nx_max - ball_r))
        ymin = max(0.0, min(ny_max, ball_r))
        ymax = min(ny_max, max(ymin, ny_max - ball_r))
        time_eps = 1e-9
        pos_eps = 1e-4
        max_reflections = 32
        iter_count = 0

        inflated_rects = [
            (
                float(rx1) - ball_r,
                float(ry1) - ball_r,
                float(rx2) + ball_r,
                float(ry2) + ball_r,
            )
            for (rx1, ry1, rx2, ry2) in self.course.solid_rects
        ]

        def resolve_penetration() -> bool:
            adjusted = False
            if pos[0] < xmin:
                pos[0] = xmin + pos_eps
                if vel[0] < 0.0:
                    vel[0] = -vel[0]
                adjusted = True
            elif pos[0] > xmax:
                pos[0] = xmax - pos_eps
                if vel[0] > 0.0:
                    vel[0] = -vel[0]
                adjusted = True

            if pos[1] < ymin:
                pos[1] = ymin + pos_eps
                if vel[1] < 0.0:
                    vel[1] = -vel[1]
                adjusted = True
            elif pos[1] > ymax:
                pos[1] = ymax - pos_eps
                if vel[1] > 0.0:
                    vel[1] = -vel[1]
                adjusted = True

            for rx1i, ry1i, rx2i, ry2i in inflated_rects:
                if rx1i <= pos[0] <= rx2i and ry1i <= pos[1] <= ry2i:
                    dl = pos[0] - rx1i
                    dr = rx2i - pos[0]
                    dtp = pos[1] - ry1i
                    db = ry2i - pos[1]
                    m = min(dl, dr, dtp, db)
                    if m == dl:
                        pos[0] = rx1i - pos_eps
                        if vel[0] > 0.0:
                            vel[0] = -abs(vel[0])
                        adjusted = True
                    elif m == dr:
                        pos[0] = rx2i + pos_eps
                        if vel[0] < 0.0:
                            vel[0] = abs(vel[0])
                        adjusted = True
                    elif m == dtp:
                        pos[1] = ry1i - pos_eps
                        if vel[1] > 0.0:
                            vel[1] = -abs(vel[1])
                        adjusted = True
                    else:
                        pos[1] = ry2i + pos_eps
                        if vel[1] < 0.0:
                            vel[1] = abs(vel[1])
                        adjusted = True
            return adjusted

        for _ in range(4):
            if not resolve_penetration():
                break

        while remaining > time_eps and iter_count < max_reflections:
            iter_count += 1
            vx = float(vel[0])
            vy = float(vel[1])
            if abs(vx) <= time_eps and abs(vy) <= time_eps:
                break

            px = float(pos[0])
            py = float(pos[1])

            hit_t = remaining + time_eps
            hit_normals: list[tuple[float, float]] = []

            def register_hit(t_candidate: float, normal: tuple[float, float]) -> None:
                nonlocal hit_t, hit_normals
                if t_candidate < -time_eps or t_candidate > remaining + time_eps:
                    return
                t = max(t_candidate, 0.0)
                if t < hit_t - time_eps:
                    hit_t = t
                    hit_normals = [normal]
                elif abs(t - hit_t) <= time_eps:
                    hit_normals.append(normal)

            if vx < -time_eps:
                t = (xmin - px) / vx
                if t >= -time_eps:
                    y_at = py + vy * t
                    if (ymin - pos_eps) <= y_at <= (ymax + pos_eps):
                        register_hit(t, (1.0, 0.0))
            if vx > time_eps:
                t = (xmax - px) / vx
                if t >= -time_eps:
                    y_at = py + vy * t
                    if (ymin - pos_eps) <= y_at <= (ymax + pos_eps):
                        register_hit(t, (-1.0, 0.0))
            if vy < -time_eps:
                t = (ymin - py) / vy
                if t >= -time_eps:
                    x_at = px + vx * t
                    if (xmin - pos_eps) <= x_at <= (xmax + pos_eps):
                        register_hit(t, (0.0, 1.0))
            if vy > time_eps:
                t = (ymax - py) / vy
                if t >= -time_eps:
                    x_at = px + vx * t
                    if (xmin - pos_eps) <= x_at <= (xmax + pos_eps):
                        register_hit(t, (0.0, -1.0))

            for rx1i, ry1i, rx2i, ry2i in inflated_rects:
                if vx > time_eps and px <= rx1i - pos_eps:
                    t = (rx1i - px) / vx
                    if t >= -time_eps:
                        y_at = py + vy * t
                        if ry1i - pos_eps <= y_at <= ry2i + pos_eps:
                            register_hit(t, (-1.0, 0.0))
                if vx < -time_eps and px >= rx2i + pos_eps:
                    t = (rx2i - px) / vx
                    if t >= -time_eps:
                        y_at = py + vy * t
                        if ry1i - pos_eps <= y_at <= ry2i + pos_eps:
                            register_hit(t, (1.0, 0.0))
                if vy > time_eps and py <= ry1i - pos_eps:
                    t = (ry1i - py) / vy
                    if t >= -time_eps:
                        x_at = px + vx * t
                        if rx1i - pos_eps <= x_at <= rx2i + pos_eps:
                            register_hit(t, (0.0, -1.0))
                if vy < -time_eps and py >= ry2i + pos_eps:
                    t = (ry2i - py) / vy
                    if t >= -time_eps:
                        x_at = px + vx * t
                        if rx1i - pos_eps <= x_at <= rx2i + pos_eps:
                            register_hit(t, (0.0, 1.0))

            if not hit_normals or hit_t >= remaining - time_eps:
                pos[0] += vx * remaining
                pos[1] += vy * remaining
                remaining = 0.0
                break

            travel = max(hit_t, 0.0)
            if travel > 0.0:
                pos[0] += vx * travel
                pos[1] += vy * travel
            remaining = max(0.0, remaining - travel)

            seen: set[tuple[float, float]] = set()
            offset_x = 0.0
            offset_y = 0.0
            for normal in hit_normals:
                key = (round(normal[0], 3), round(normal[1], 3))
                if key in seen:
                    continue
                seen.add(key)
                nx, ny = normal
                dot = float(vel[0]) * nx + float(vel[1]) * ny
                if dot >= 0.0:
                    continue
                vel[0] -= 2.0 * dot * nx
                vel[1] -= 2.0 * dot * ny
                offset_x += nx
                offset_y += ny

            offset_norm = math.hypot(offset_x, offset_y)
            if offset_norm > 0.0:
                pos[0] += (offset_x / offset_norm) * pos_eps
                pos[1] += (offset_y / offset_norm) * pos_eps

            resolve_penetration()

        if remaining > time_eps:
            pos[0] += float(vel[0]) * remaining
            pos[1] += float(vel[1]) * remaining

        resolve_penetration()

        pos[0] = float(min(max(pos[0], xmin), xmax))
        pos[1] = float(min(max(pos[1], ymin), ymax))
        return pos

    def _shoot(self, kvec_cpu):
        # Movement speed scaling: modify momentum while keeping the quantum dt stable
        self._stop_playback()
        if self.show_interference:
            self.viz.set_interference_visible(True)
            if self._interference_profile is not None:
                self.viz.update_interference_pattern(np.zeros_like(self._interference_profile))
        old_dt = self.cfg.dt
        movement_factor = float(max(0.1, getattr(self, '_movement_speed_factor', 1.0)))
        speed_boost = movement_factor
        dt_eff = float(old_dt)
        dt_scale = 1.0
        dt_eff = old_dt  # keep classical integrator stable; wave safety is handled elsewhere

        simulate_classical = self._mode_allows_classical()
        simulate_wave = self._mode_allows_quantum()

        if not simulate_classical and not simulate_wave:
            return

        shot_stop_mode = str(getattr(self.cfg, 'shot_stop_mode', 'time')).lower()
        friction_mode = shot_stop_mode == 'friction'
        sink_mode = str(getattr(self.cfg, 'sink_rule', 'prob_threshold')).lower()
        if friction_mode:
            friction_coeffs = (
                float(getattr(self.cfg, 'shot_friction_linear', 0.337)),
                float(getattr(self.cfg, 'shot_friction_quadratic', 0.25)),
                float(getattr(self.cfg, 'shot_friction_cubic', 0.413)),
            )
            friction_min_scale = float(max(0.0, getattr(self.cfg, 'shot_friction_min_scale', 0.01)))
        else:
            friction_coeffs = None
            friction_min_scale = 0.0

        self._wavefront_profile = str(getattr(self.cfg, 'wave_initial_profile', 'packet')).lower()
        self._wavefront_active = self._wavefront_profile == 'front' and simulate_wave

        # temp exponents for this shot
        self._ensure_exponents(dt_eff)

        self._abort_shot_requested = False

        try:
            sunk_prob = 0.0
            base_steps = self.base_steps_per_shot * (self.cfg.perf_steps_factor if self.perf_mode else 1.0)
            if dt_scale > 0.0:
                base_steps *= dt_scale
            steps = int(max(1, round(base_steps)))
            shot_limit = getattr(self.cfg, 'shot_time_limit', None)
            if shot_limit is not None and shot_limit > 0.0:
                limit_steps = int(math.ceil(shot_limit / max(dt_eff, 1e-9)))
                max_steps = int(max(1, getattr(self.cfg, 'max_steps_per_shot', limit_steps)))
                steps = int(max(1, min(limit_steps, max_steps)))
            draw_every = self.perf_draw_every if self.perf_mode else self.base_draw_every
            if self.perf_mode and dt_scale > 1.0:
                draw_every = max(1, int(round(draw_every * dt_scale)))
            sigma = self.cfg.perf_sigma0 if self.perf_mode else self.cfg.sigma0

            if self.cfg.flags.adaptive_draw:
                cw, ch = self.viz.fig.canvas.get_width_height()
                pixel = max(1, cw * ch)
                scale_adapt = pixel / (900 * 600)
                draw_every = max(
                    1, int(round(draw_every * scale_adapt * (30.0 / self.cfg.target_fps))))

            self._overlay_compute_during_shot = bool(self.show_info)
            self._pending_info_overlay = False
            self._pending_measure_after_shot = False
            self.shot_in_progress = True
            self.game_over = False
            if self._multiple_shots_enabled:
                self._shot_count += 1
                self._update_shot_counter()
            self.viz.clear_classical_overlay()
            self.viz.clear_wave_overlay(simulate_wave)
            self.viz.measure_marker.set_visible(False)
            self.viz.measure_point.set_visible(False)
            self._last_density_cpu = None
            self._last_measure_xy = None
            self._last_measure_prob = None
            self._last_ex = None
            self._last_ey = None

            xp = self.be.xp if simulate_wave else None
            base_expV_half = self.course.expV_half
            base_expK = self.course.expK
            use_friction_lut = False
            V_operator_cache = tmp_v_operator = k2_cache = tmp_k2 = None
            if simulate_wave and friction_mode:
                if self._perf_enabled:
                    self._build_friction_lookup(base_expV_half, base_expK, dt_eff, xp)
                    use_friction_lut = self._perf_friction_expV is not None
                if not use_friction_lut:
                    V_operator_cache = self.course.V_operator.astype(self.c64, copy=False)
                    k2_cache = (-1j * self.k2).astype(self.c64, copy=False)
                    tmp_v_operator = xp.empty_like(V_operator_cache, dtype=self.c64)
                    tmp_k2 = xp.empty_like(k2_cache, dtype=self.c64)

            friction_prev_scale = 1.0
            friction_stop_flag = False

            kvec_eff = np.array(kvec_cpu, dtype=np.float32) * speed_boost
            kmax = float(self.be.to_cpu(self.k_max))
            kmag = float(np.linalg.norm(kvec_eff))
            if kmag > kmax:
                kvec_eff *= (kmax / (kmag + 1e-12))
                kmag = float(np.linalg.norm(kvec_eff))

            psi = None
            wave_xs = []
            wave_ys = []
            fft_plan = None

            if simulate_wave:
                X = np.arange(self.Nx, dtype=np.float32)
                Y = np.arange(self.Ny, dtype=np.float32)
                Xg, Yg = np.meshgrid(X, Y, indexing='xy')
                profile = str(getattr(self.cfg, 'wave_initial_profile', 'packet')).lower()
                self._wavefront_profile = profile
                x0, y0 = self.ball_pos
                if profile == 'front':
                    sigma_y = float(max(1e-3, getattr(self.cfg, 'wavefront_sigma_y', sigma * 2.0)))
                    trans_len = float(max(1e-3, getattr(self.cfg, 'wavefront_transition_len', sigma)))
                    sigma_forward = float(max(1e-3, getattr(self.cfg, 'wavefront_sigma_forward', trans_len)))
                    if kmag > 1e-6:
                        dir_vec = kvec_eff / kmag
                    else:
                        dir_vec = np.array([1.0, 0.0], dtype=np.float32)
                    dir_vec = dir_vec.astype(np.float32)
                    self._wavefront_dir = dir_vec
                    self._wavefront_active = True
                    self._wavefront_kmag = kmag
                    perp = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float32)
                    s = (Xg - x0) * dir_vec[0] + (Yg - y0) * dir_vec[1]
                    p = (Xg - x0) * perp[0] + (Yg - y0) * perp[1]
                    offset = float(max(0.0, getattr(self.cfg, 'wavefront_start_offset', 0.0)))
                    s_shift = s + offset
                    s_pos = np.maximum(s_shift, 0.0)
                    forward_gauss = np.exp(-(s_pos ** 2) /
                                           (2 * (sigma_forward ** 2))).astype(np.float32)
                    transition = 0.5 * (1.0 + np.tanh(s_shift / trans_len)).astype(np.float32)
                    envelope_dir = forward_gauss * transition
                    envelope_perp = np.exp(-(p ** 2) /
                                           (2 * (sigma_y ** 2))).astype(np.float32)
                    gauss = np.clip(envelope_dir * envelope_perp, 0.0, None)
                else:
                    self._wavefront_active = False
                    self._wavefront_kmag = 0.0
                    self._wavefront_dir = np.array([1.0, 0.0], dtype=np.float32)
                    gauss = np.exp(-((Xg - x0) ** 2 + (Yg - y0) ** 2) /
                                   (2 * sigma ** 2)).astype(np.float32)

                plane = np.exp(
                    1j * (kvec_eff[0] * Xg + kvec_eff[1] * Yg)).astype(np.complex64)
                gauss = np.clip(gauss, 0.0, None)
                psi = self.be.to_xp(gauss, np.float32).astype(
                    np.complex64) * self.be.to_xp(plane, np.complex64)
                psi /= xp.sqrt(xp.sum(xp.abs(psi) ** 2) + 1e-12)
                psi = self.course.apply_edge_boundary(psi)
                fft_plan = self._ensure_fft_plan(psi) if self._perf_enabled else None
            else:
                Xg = Yg = None
                fft_plan = None

            if simulate_classical:
                x0, y0 = self.ball_pos
                c_pos = np.array([float(x0), float(y0)], dtype=float)
                c_v = np.array(
                    [float(kvec_eff[0]), float(kvec_eff[1])], dtype=float)
                c_speed = np.linalg.norm(c_v) + 1e-9
                c_dtc = min(0.5, 0.6 / c_speed)
                c_t = 0.0
                class_xs = [c_pos[0]] if self._render_paths else []
                class_ys = [c_pos[1]] if self._render_paths else []
                self.viz.class_marker.center = (c_pos[0], c_pos[1])
                self.viz.class_marker.set_visible(True)
            else:
                class_xs = []
                class_ys = []
                c_pos = None
                c_v = np.zeros(2, dtype=float)
                c_dtc = 0.0
                c_t = 0.0

            sunk = False
            renorm_every = 40
            abort_reason = None  # "time" or "hole"
            aborted = False

            n = 0
            while n < steps:
                if self._abort_shot_requested or not self.shot_in_progress:
                    aborted = True
                    if abort_reason is None:
                        abort_reason = "reset"
                    break
                if friction_mode:
                    speed_scale = self._friction_speed_scale(n, steps, friction_coeffs)
                    speed_scale = max(0.0, speed_scale)
                else:
                    speed_scale = 1.0

                if friction_mode and self._wavefront_active:
                    self._wavefront_kmag = kmag * speed_scale

                if simulate_wave:
                    if self._perf_enabled and not friction_mode:
                        psi, consumed = self._maybe_burst_drift(psi, base_expK, dt_eff, steps - n, fft_plan)
                        if consumed > 1:
                            psi = self.course.apply_edge_boundary(psi)
                            n += consumed
                            continue
                    if friction_mode:
                        if use_friction_lut:
                            idx = self._lookup_friction_index(speed_scale)
                            expV_half_eff = self._perf_friction_expV[idx]
                            expK_eff = self._perf_friction_expK[idx]
                        else:
                            dt_local = dt_eff * float(speed_scale)
                            if speed_scale >= 1.0 - 1e-6:
                                expV_half_eff = base_expV_half
                                expK_eff = base_expK
                            else:
                                half_dt = dt_local * 0.5
                                xp.multiply(V_operator_cache, half_dt, out=tmp_v_operator)
                                xp.exp(tmp_v_operator, out=tmp_v_operator)
                                xp.multiply(k2_cache, half_dt, out=tmp_k2)
                                xp.exp(tmp_k2, out=tmp_k2)
                                expV_half_eff = tmp_v_operator
                                expK_eff = tmp_k2
                    else:
                        expV_half_eff = base_expV_half
                        expK_eff = base_expK

                    if not (friction_mode and speed_scale <= friction_min_scale):
                        plan_ctx = fft_plan if (self._perf_enabled and self.be.USE_GPU) else None
                        psi = step_wave(
                            psi,
                            expV_half_eff,
                            expK_eff,
                            self.be.fft2,
                            self.be.ifft2,
                            inplace=self.cfg.flags.inplace_step,
                            plan=plan_ctx,
                        )
                    psi = self.course.apply_edge_boundary(psi)

                if simulate_classical:
                    if friction_mode:
                        if friction_prev_scale <= 1e-9:
                            ratio = 0.0
                        else:
                            ratio = speed_scale / max(friction_prev_scale, 1e-9)
                        c_v *= ratio
                        c_speed = np.linalg.norm(c_v) + 1e-9
                        c_dtc = min(0.5, 0.6 / c_speed)
                    t_target = (n + 1) * dt_eff
                    if n == 0:
                        c_t = 0.0
                    while c_t < t_target and self.viz.class_marker.get_visible():
                        if self._abort_shot_requested or not self.shot_in_progress:
                            aborted = True
                            if abort_reason is None:
                                abort_reason = "reset"
                            break
                        step = min(c_dtc, t_target - c_t)
                        c_pos = self._advance_classical_segment(c_pos, c_v, step)
                        self.ball_pos[0] = float(c_pos[0])
                        self.ball_pos[1] = float(c_pos[1])
                        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])
                        if self._render_paths:
                            class_xs.append(c_pos[0])
                            class_ys.append(c_pos[1])
                        c_t += step

                        dx = c_pos[0] - self.hole_center[0]
                        dy = c_pos[1] - self.hole_center[1]
                        if dx * dx + dy * dy <= (self.cfg.hole_r ** 2):
                            self.viz.class_marker.center = (c_pos[0], c_pos[1])
                            sunk = True
                            abort_reason = "hole"
                            self.viz.show_messages(
                                wave_hit=False, ball_hit=True)
                            break

                if sunk:
                    break

                if simulate_wave and (n % renorm_every) == 0:
                    psi /= xp.sqrt(xp.sum(xp.abs(psi) ** 2) + 1e-12)

                if (n % draw_every) == 0 or n == steps - 1:
                    dens = None
                    if simulate_wave:
                        dens = xp.abs(psi) ** 2
                        display_dens = self._build_display_density(dens, psi)
                        self._present_density_frame(display_dens, plot_wave=True)
                        if self.show_interference:
                            self._update_interference_pattern(density=dens)
                    if simulate_classical:
                        if self._render_paths and class_xs:
                            xs = class_xs[::self.path_decim] if (
                                self.cfg.flags.path_decimation and self.path_decim > 1) else class_xs
                            ys = class_ys[::self.path_decim] if (
                                self.cfg.flags.path_decimation and self.path_decim > 1) else class_ys
                            self.viz.class_path_line.set_data(xs, ys)
                            self.viz.class_path_line.set_visible(True)
                        elif not self._render_paths:
                            self.viz.class_path_line.set_visible(False)
                        if c_pos is not None:
                            self.viz.class_marker.center = (c_pos[0], c_pos[1])
                        self.viz.class_marker.set_visible(True)
                    if simulate_wave:
                        ex, ey = compute_expectation(
                            self.Xgrid, self.Ygrid, dens, xp, self.be.to_cpu)
                        self._last_ex, self._last_ey = ex, ey
                        if self._overlay_compute_during_shot and self.show_info and (n % self.cfg.overlay_every == 0):
                            ex2, ey2, a1, b1, ang = covariance_ellipse(
                                self.Xgrid,
                                self.Ygrid,
                                dens,
                                xp,
                                self.be.to_cpu,
                            )
                            self.viz.update_overlay_from_stats(
                                ex2, ey2, a1, b1, ang, show=True)
                        if self._render_paths:
                            wave_xs.append(ex)
                            wave_ys.append(ey)
                            wx = wave_xs[::self.path_decim] if (
                                self.cfg.flags.path_decimation and self.path_decim > 1) else wave_xs
                            wy = wave_ys[::self.path_decim] if (
                                self.cfg.flags.path_decimation and self.path_decim > 1) else wave_ys
                            self.viz.wave_path_line.set_data(wx, wy)
                            self.viz.wave_path_line.set_visible(True)
                        else:
                            self.viz.wave_path_line.set_visible(False)
                        if sink_mode == "prob_threshold":
                            p_in = float(self.be.to_cpu(
                                (dens[self.course.hole_mask]).sum()))
                            if p_in > float(self.cfg.sink_prob_threshold):
                                sunk = True
                if aborted:
                    break
                if not simulate_wave:
                        if self.cfg.flags.blitting:
                            self.viz._blit_draw()
                        else:
                            self.viz.fig.canvas.draw_idle()
                        self._notify_frame_listeners()

                if friction_mode:
                    friction_prev_scale = speed_scale
                    if speed_scale <= friction_min_scale:
                        friction_stop_flag = True

                if friction_mode and friction_stop_flag and not sunk:
                    abort_reason = abort_reason or "friction"
                    break

                if (self.cfg.shot_time_limit is not None) and (((n + 1) * dt_eff) >= self.cfg.shot_time_limit):
                    abort_reason = "time"
                    break

                if self.cfg.flags.blitting:
                    canvas = self.viz.fig.canvas
                    flush_events = getattr(canvas, "flush_events", None)
                    if callable(flush_events):
                        try:
                            flush_events()
                        except Exception:
                            pass
                else:
                    plt.pause(0.0001)

                if self._abort_shot_requested or not self.shot_in_progress:
                    aborted = True
                    if abort_reason is None:
                        abort_reason = "reset"
                    break

                n += 1

            if aborted:
                self._abort_shot_requested = False
                self._last_density_cpu = None
                self._last_measure_xy = None
                self._last_measure_prob = None
                self._pending_info_overlay = False
                self._pending_measure_after_shot = False
            else:
                if simulate_wave:
                    self._last_density_cpu = self.be.to_cpu(
                        (xp.abs(psi) ** 2)).astype(np.float32)
                    if self._wavefront_active:
                        self._last_phase_cpu = np.angle(self.be.to_cpu(psi))
                    else:
                        self._last_phase_cpu = None
                    self._update_interference_pattern()
                    if abort_reason != "hole":
                        sunk_prob = float(
                            self._last_density_cpu[self.course.hole_mask_cpu].sum())
                        if sink_mode == "prob_threshold":
                            sunk = sunk or (sunk_prob > float(self.cfg.sink_prob_threshold))
                else:
                    self._last_density_cpu = None

                measured_xy = None
                measured_local_prob = None
                if simulate_wave and self.cfg.quantum_measure:
                    measured_xy = self._do_end_measurement()
                    measured_local_prob = self._last_measure_prob
                    if (sink_mode == "measurement") and (measured_xy is not None) and (abort_reason != "hole"):
                        mx, my = measured_xy
                        dx = mx - self.hole_center[0]
                        dy = my - self.hole_center[1]
                        in_hole = (dx * dx + dy * dy) <= (self.cfg.hole_r ** 2)
                        min_prob = float(getattr(self.cfg, "measurement_sink_min_prob", 1e-3))
                        local_prob = float(measured_local_prob) if measured_local_prob is not None else 0.0
                        effective_prob = max(local_prob, sunk_prob)
                        if in_hole and (effective_prob >= min_prob):
                            sunk = True
                else:
                    self._last_measure_xy = None
                    self._last_measure_prob = None

            if simulate_wave and self._last_density_cpu is not None and self.show_info:
                dens_xp = self.be.to_xp(self._last_density_cpu, np.float32)
                xp = self.be.xp
                ex2, ey2, a1, b1, ang = covariance_ellipse(
                    self.Xgrid, self.Ygrid, dens_xp, xp, self.be.to_cpu
                )
                self.viz.update_overlay_from_stats(ex2, ey2, a1, b1, ang, show=True)
                if self._last_measure_xy is not None:
                    mx, my = self._last_measure_xy
                    self.viz.set_measure_point(mx, my, True)
                self.viz.set_wave_path_label(True)
                self._pending_info_overlay = False
            elif not self.show_info:
                self.viz.set_info_visibility(False)
                self.viz.set_wave_path_label(False)

            self.viz.update_title(self._title_text(), sunk=bool(sunk))
            if simulate_classical:
                self.viz.class_marker.set_visible(True)
            else:
                self.viz.class_marker.set_visible(False)
            if simulate_wave and (self._last_ex is not None):
                self.viz.wave_end_marker.center = (
                    self._last_ex, self._last_ey)
                self.viz.wave_end_marker.set_visible(True)
            else:
                self.viz.wave_end_marker.set_visible(False)

            if self.cfg.flags.blitting:
                self.viz._blit_draw()
            else:
                self.viz.fig.canvas.draw_idle()

            self.game_over = True
            self.shot_in_progress = False
            self._overlay_compute_during_shot = False
            self._pending_info_overlay = False
            if self._pending_measure_after_shot:
                self._measure_now()
            self._pending_measure_after_shot = False

        finally:
            # restore dt exponents
            if self.tracker:
                self._update_tracker_reference()
            self.course.update_exponents(old_dt, self.k2, self.c64)
        if self._multiple_shots_enabled:
            if sunk:
                self._reset(ball_only=False)
            else:
                self.game_over = False
                self._update_shot_counter()
    # ----- measurement helpers

    def _do_end_measurement(self):
        if self._last_density_cpu is None or not self._mode_allows_measure():
            self.viz.measure_point.set_visible(False)
            self.viz.measure_marker.set_visible(False)
            self._last_measure_xy = None
            return None

        sink_mode = str(getattr(self.cfg, 'sink_rule', 'prob_threshold')).lower()
        use_boost = self.cfg.boost_hole_probability and sink_mode != 'measurement'
        beta = self.cfg.boost_hole_probability_factor if use_boost else 0.0
        mx, my = sample_from_density(self._last_density_cpu, self.cfg.measure_gamma,
                                     self.course.hole_mask_cpu, beta, rng=self.rng)
        self._last_measure_xy = (mx, my)
        iy = int(np.clip(round(my), 0, self.Ny - 1))
        ix = int(np.clip(round(mx), 0, self.Nx - 1))
        self._last_measure_prob = float(self._last_density_cpu[iy, ix])
        if self.show_info:
            self.viz.set_measure_point(mx, my, True)
        else:
            self.viz.measure_point.set_visible(False)
        self.viz.measure_marker.set_visible(False)
        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        return mx, my

    def _measure_now(self):
        if not self._mode_allows_measure():
            return
        if self._last_density_cpu is None:
            return
        if self.shot_in_progress and not self._overlay_compute_during_shot:
            self._pending_measure_after_shot = True
            if not self.show_info:
                self.show_info = True
            self._pending_info_overlay = True
            self.viz.set_info_visibility(False)
            self.viz.measure_point.set_visible(False)
            self.viz.measure_marker.set_visible(False)
            self.viz.set_wave_path_label(False)
            return
        self._pending_measure_after_shot = False
        self._pending_info_overlay = False
        if not self.show_info:
            self.show_info = True

        self.viz.set_wave_path_label(True)
        sink_mode = str(getattr(self.cfg, 'sink_rule', 'prob_threshold')).lower()
        use_boost = self.cfg.boost_hole_probability and sink_mode != 'measurement'
        beta = self.cfg.boost_hole_probability_factor if use_boost else 0.0
        mx, my = sample_from_density(self._last_density_cpu, self.cfg.measure_gamma,
                                     self.course.hole_mask_cpu, beta, rng=self.rng)
        self._last_measure_xy = (mx, my)
        iy = int(np.clip(round(my), 0, self.Ny - 1))
        ix = int(np.clip(round(mx), 0, self.Nx - 1))
        self._last_measure_prob = float(self._last_density_cpu[iy, ix])

        dx = mx - self.hole_center[0]
        dy = my - self.hole_center[1]
        in_hole = (dx * dx + dy * dy) <= (self.cfg.hole_r ** 2)
        self.viz.show_messages(wave_hit=in_hole, ball_hit=None)
        self.viz.set_measure_point(mx, my, True)

        # update overlay ellipses immediately
        xp = self.be.xp
        dens_xp = self.be.to_xp(self._last_density_cpu, np.float32)
        ex, ey, a1, b1, ang = covariance_ellipse(
            self.Xgrid, self.Ygrid, dens_xp, xp, self.be.to_cpu)
        self.viz.update_overlay_from_stats(ex, ey, a1, b1, ang, show=True)

        # optional auto-increment bias
        if (self.cfg.boost_hole_probability and
            self.cfg.boost_hole_probability_autoincrement_on_measure and
                self.cfg.boost_hole_probability_increment > 0.0):
            self.cfg.boost_hole_probability_factor = float(
                min(1.0, self.cfg.boost_hole_probability_factor +
                    self.cfg.boost_hole_probability_increment)
            )

        if self.cfg.flags.blitting:
            self.viz._blit_draw()
        else:
            self.viz.fig.canvas.draw_idle()
