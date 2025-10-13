from __future__ import annotations
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
from .config import GameConfig
from .backends import Backend
from .course import Course
from .physics import (
    step_wave, prepare_frame, compute_expectation, covariance_ellipse, sample_from_density
)
from .visuals import Visuals


class QuantumMiniGolfGame:
    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.be = Backend()
        if self.be.USE_GPU:
            self.cfg.flags.gpu_viz = True

        # numeric dtypes & grids
        self.f32 = np.float32
        self.c64 = np.complex64

        # course
        self.course = Course(cfg, self.be)
        self.Nx, self.Ny = self.course.Nx, self.course.Ny

        # k-space (CPU -> XP)
        self.k2, self.k_max = self.be.build_kgrid(
            self.Nx, self.Ny, cfg.dx, cfg.dy)

        # grids for expectations
        Xc, Yc = np.meshgrid(np.arange(self.Nx, dtype=np.float32),
                             np.arange(self.Ny, dtype=np.float32), indexing='xy')
        self.Xgrid = self.be.to_xp(Xc, np.float32)
        self.Ygrid = self.be.to_xp(Yc, np.float32)

        # exponents for current dt
        self.course.update_exponents(cfg.dt, self.k2, self.c64)

        # positions
        start_x = int(
            round(max(8, min(self.Nx - 8, self.Nx * self.cfg.ball_start_x_frac))))
        self.hole_center = self.course.hole_center.copy()
        self.ball_pos = np.array(
            [float(start_x), self.Ny / 2], dtype=np.float32)

        # visuals
        self.viz = Visuals(self.Nx, self.Ny, self.hole_center,
                           self.cfg.hole_r, self.cfg.flags, self.cfg)
        self.viz.set_course_patches(self.course.course_patches)
        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])
        self.viz.update_title(self._title_text())
        self.tracker = None
        self.tracker_cfg = None
        self._tracker_timer = None
        if getattr(self.cfg, 'use_tracker', False):
            try:
                self.tracker_cfg = TrackerConfig(
                    show_debug_window=self.cfg.tracker_debug_window,
                    crop_x1=getattr(self.cfg, 'tracker_crop_x1', None),
                    crop_x2=getattr(self.cfg, 'tracker_crop_x2', None),
                    crop_y1=getattr(self.cfg, 'tracker_crop_y1', None),
                    crop_y2=getattr(self.cfg, 'tracker_crop_y2', None),
                )
                self.tracker = TrackerManager(self.tracker_cfg)
                self.tracker.start()
            except Exception as exc:
                print(f'Tracker disabled: {exc}')
                self.tracker = None
            else:
                self._start_tracker_poll()
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

        # swing detection
        self.cursor_inside = False
        self.indicator_pos = None
        self.last_mouse_pos = None
        self.last_mouse_t = None

        self.ds_factor = int(max(1, self.cfg.display_downsample_factor))
        self.path_decim = int(max(1, self.cfg.path_decimation_stride))

        # perf-mode values (you can expose a toggle later)
        self.perf_mode = False
        self.perf_draw_every = max(1, int(self.cfg.draw_every * 2))
        self.base_steps_per_shot = self.cfg.steps_per_shot
        self.base_draw_every = self.cfg.draw_every

        # RNG for sampling
        self.rng = np.random.default_rng()
        self._gpu_rgba_buffer = None

        # movement-speed tuning (store baselines so slider changes keep swing tuning consistent)
        self._movement_slider_bounds = (0.5, 15.0)
        self._base_kmin_frac = float(getattr(cfg, 'kmin_frac', 0.15))
        self._base_kmax_frac = float(getattr(cfg, 'kmax_frac', 0.90))
        self._base_swing_power_scale = float(getattr(cfg, 'swing_power_scale', 0.05))
        self._base_impact_min_speed = float(getattr(cfg, 'impact_min_speed', 20.0))
        # direct multiplier applied to post-shot motion
        self._movement_speed_factor = float(max(self._movement_slider_bounds[0], getattr(cfg, 'movement_speed_scale', 1.0)))
        self._apply_movement_speed_tuning(initial=True)

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
        raw_value = float(getattr(self.cfg, 'movement_speed_scale', 1.0))
        if not math.isfinite(raw_value):
            raw_value = 1.0
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
            s_pos = np.maximum(s, 0.0)
            forward_gauss = np.exp(-(s_pos ** 2) /
                                   (2.0 * sigma_forward * sigma_forward)).astype(np.float32)
            transition = 0.5 * (1.0 + np.tanh(s / trans_len)).astype(np.float32)
            envelope_dir = forward_gauss * transition
            envelope_perp = np.exp(-(p ** 2) /
                                   (2.0 * sigma_y * sigma_y)).astype(np.float32)
            amp = np.clip(envelope_dir * envelope_perp, 0.0, None)
            kmax = float(self.be.to_cpu(self.k_max))
            kmag = 0.5 * (self.cfg.kmin_frac + self.cfg.kmax_frac) * kmax
            stripes = 0.5 * (1.0 + np.cos(kmag * s)).astype(np.float32)
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

    def _start_tracker_poll(self):
        if not self.tracker:
            return
        self._update_tracker_reference()
        self._tracker_timer = self.viz.fig.canvas.new_timer(interval=33)
        self._tracker_timer.add_callback(self._poll_tracker)
        self._tracker_timer.start()

    def _on_close(self, _event):
        if self._tracker_timer is not None:
            try:
                self._tracker_timer.stop()
            except Exception:
                pass
            self._tracker_timer = None
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
        x = px[0] / self.tracker_cfg.frame_width * self.Nx
        y = (1.0 - px[1] / self.tracker_cfg.frame_height) * self.Ny
        return float(x), float(y)

    def _tracker_dir_to_game(self, direction_px: tuple[float, float]) -> np.ndarray:
        if not self.tracker_cfg:
            return np.array(direction_px, dtype=float)
        scale_x = self.Nx / self.tracker_cfg.frame_width
        scale_y = self.Ny / self.tracker_cfg.frame_height
        return np.array([direction_px[0] * scale_x, -direction_px[1] * scale_y], dtype=float)

    def _poll_tracker(self):
        if not self.tracker:
            return
        state = self.tracker.get_state()
        self._update_tracker_reference()
        visible = state.visible and state.center_px is not None and state.direction_px is not None
        if visible:
            center = self._tracker_px_to_game(state.center_px)
            dir_game = self._tracker_dir_to_game(state.direction_px)
            norm = np.linalg.norm(dir_game)
            if norm < 1e-6:
                visible = False
            else:
                dir_unit = dir_game / norm
                angle_deg = math.degrees(math.atan2(dir_unit[1], dir_unit[0]))
                length_game = (self.cfg.tracker_length_scale *
                               self.tracker_cfg.putter_length_px / self.tracker_cfg.frame_width) * self.Nx
                thickness_game = (self.cfg.tracker_thickness_scale *
                                  self.tracker_cfg.putter_thickness_px / self.tracker_cfg.frame_height) * self.Ny
                self.viz.update_putter_overlay(
                    center, length_game, thickness_game, angle_deg, True)
        if not visible:
            self.viz.update_putter_overlay((0.0, 0.0), 0.0, 0.0, 0.0, False)
        hits = self.tracker.pop_hits()
        for hit in hits:
            self._handle_tracker_hit(hit)

    def _handle_tracker_hit(self, hit):
        if self.shot_in_progress or self.game_over:
            return
        dir_game = self._tracker_dir_to_game(hit.direction_px)
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
        elif key == 'r':
            if not self._restore_normal_state():
                self._reset()
        elif key == 'tab':
            self._toggle_map()
        elif key == 'c':
            self._cycle_mode()
        elif key == 'm':
            if self._mode_allows_measure():
                self._measure_now()
        elif key == 'i':
            if self._mode_allows_quantum():
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
        elif key == 'u':
            self._toggle_config_panel()
        elif key == 'l':
            self._toggle_interference_pattern()
        elif key == 'd':
            self._play_recording()
        elif key == 'h':
            self._print_hotkey_help()

    # ----- map / reset / info
    def _toggle_map(self):
        if self.shot_in_progress:
            return
        order = ['double_slit', 'single_slit', 'single_wall', 'no_obstacle']
        try:
            idx = order.index(self.cfg.map_kind)
        except ValueError:
            idx = 0
        self._switch_map(order[(idx + 1) % len(order)])

    def _switch_map(self, kind):
        if self.shot_in_progress:
            return
        self.course.set_map(kind)
        self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
        self.viz.set_course_patches(self.course.course_patches)
        self._reset(ball_only=False)
        self.viz.update_title(self._title_text())

    def _toggle_info_overlay(self):
        if not self._mode_allows_quantum():
            return
        self.show_info = not self.show_info
        if not self.show_info:
            self.viz.set_info_visibility(False)
            self.viz.set_wave_path_label(False)
            self.viz.show_messages(False, False)
            self.viz.fig.canvas.draw_idle()
        else:
            # draw immediately if density known
            if self._last_density_cpu is not None and self._mode_allows_quantum():
                ex, ey, a1, b1, ang = covariance_ellipse(
                    self.Xgrid, self.Ygrid,
                    self.be.to_xp(self._last_density_cpu, np.float32),
                    self.be.xp, self.be.to_cpu
                )
                self.viz.update_overlay_from_stats(
                    ex, ey, a1, b1, ang, show=True)
                if self._last_measure_xy is not None:
                    mx, my = self._last_measure_xy
                    self.viz.set_measure_point(mx, my, True)
            self.viz.set_wave_path_label(True)
            self.viz.hole_msg.set_visible(False)
            self.viz.hole_msg_ball.set_visible(False)
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
            self.cfg.single_wall_thickness_factor = float(np.clip(preset['single_wall_thickness_factor'], 0.05, 5.0))
            if self._config_panel_active and 'wall' in self._panel_sliders:
                self._panel_updating = True
                try:
                    self._panel_sliders['wall'].set_val(self.cfg.single_wall_thickness_factor)
                finally:
                    self._panel_updating = False

        if 'map' in preset:
            self.cfg.map_kind = preset['map']
            self.course.set_map(self.cfg.map_kind)
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
        panel_state = 'enabled' if getattr(self.cfg, 'show_control_panel', True) else 'disabled'

        entries = [
            ("q", "Quit the game", None),
            ("r", "Reset ball and wave", None),
            ("tab", "Cycle obstacle map", self.cfg.map_kind),
            ("c", "Cycle display mode", self.mode),
            ("m", "Force quantum measurement", measurement_state),
            ("i", "Toggle info overlay", info_state),
            ("#", "Cycle quantum demo course", course_demo_state),
            ("-", "Cycle advanced showcase course", course_show_state),
            ("b", "Toggle edge boundary reflect/absorb", self.cfg.edge_boundary),
            ("w", "Toggle wave initial profile packet/front", getattr(self.cfg, 'wave_initial_profile', 'packet')),
            ("t", "Toggle shot stop mode time/friction", getattr(self.cfg, 'shot_stop_mode', 'time')),
            ("g", "Toggle mouse swing control", mouse_state),
            ("u", "Toggle control panel window", 'open' if self._config_panel_active else 'closed'),
            ("panel", "Control panel window (config.show_control_panel)", panel_state),
            ("h", "Show this hotkey list", None),
        ]
        lines = []
        for key, desc, state in entries:
            if state is not None:
                lines.append(f"  {key:<3} - {desc} (current: {fmt_state(state)})")
            else:
                lines.append(f"  {key:<3} - {desc}")
        print("\nHotkeys:\n" + "\n".join(lines) + "\n")

    def _announce_switch(self, key: str, label: str, old, new):
        BLUE = "\033[34m\033[4m"
        GREEN = "\033[32m\033[4m"
        RESET = "\033[0m"
        print(f"[{key}] {label}: {BLUE}{old}{RESET} \u2192 {GREEN}{new}{RESET}")

    # ----- playback helpers
    def _play_recording(self, path: str | Path = None):
        if path is None:
            path = Path(os.getcwd()) / "MinigolfDemo.mp4"

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

    def _activate_config_panel(self):
        if self._config_panel_active or Slider is None:
            if Slider is None:
                print('[control panel] Matplotlib Slider widgets unavailable; control panel disabled.')
            return

        fig = plt.figure(figsize=(5.0, 7.2))
        fig.patch.set_facecolor('black')
        try:
            fig.canvas.manager.set_window_title('Quantum Mini-Golf - Control Panel')
        except Exception:
            pass

        self._panel_fig = fig
        self._panel_close_cid = fig.canvas.mpl_connect('close_event', self._on_panel_close)

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

        text_ax = fig.add_axes([0.08, 0.70, 0.84, 0.22])
        text_ax.axis('off')
        hotkeys_text = [
            'Hotkeys',
            ' q  quit      r  reset',
            ' tab maps     c  modes',
            ' m  measure   i  info',
            ' #  demo      -  showcase',
            ' b  boundary  w  wave',
            ' t  stop mode g  mouse',
            ' u  panel     h  help',
            ' esc close',
        ]
        y = 0.94
        for line in hotkeys_text:
            text_ax.text(0.0, y, line, color='#dddddd', fontsize=8.5, ha='left', va='top', transform=text_ax.transAxes)
            y -= 0.135
        self._panel_axes_list.append(text_ax)

        stats_text = bg_ax.text(0.08, 0.56, '', color='#bbbbbb', fontsize=8.5, ha='left', va='top', transform=bg_ax.transAxes)
        self._panel_elements['stats_text'] = stats_text

        slider_left = 0.18
        slider_width = 0.64
        slider_height = 0.045
        slider_gap = 0.06
        slider_y = 0.46

        slider_specs = [
            ('lin',  'Friction Lin',   0.0, 0.9,  float(self.cfg.shot_friction_linear),     0.01, '#1f77ff', self._on_friction_slider_change),
            ('quad', 'Friction Quad',  0.0, 0.9,  float(self.cfg.shot_friction_quadratic),  0.01, '#1f77ff', self._on_friction_slider_change),
            ('min',  'Friction Min',   0.0, 0.2,  float(self.cfg.shot_friction_min_scale),  0.005,'#ff7f0e', self._on_min_scale_change),
            ('sink', 'Sink Prob',      0.05,0.5,  float(self.cfg.sink_prob_threshold),      0.01, '#2ca02c', self._on_sink_threshold_change),
            ('boost','Boost Increment',0.0, 0.3,  float(self.cfg.boost_hole_probability_increment),0.005,'#d62728', self._on_boost_increment_change),
            ('move', 'Move Speed',     0.5, 15.0, float(self.cfg.movement_speed_scale),    0.05, '#9467bd', self._on_movement_speed_change),
            ('time', 'Shot Time',      0.0,500.0,float(self.cfg.shot_time_limit or 0.0),    10.0, '#8c564b', self._on_shot_time_limit_change),
            ('wall', 'Wall Thickness', 0.05,5.0,  float(getattr(self.cfg, 'single_wall_thickness_factor', 1.0)), 0.05, '#17becf', self._on_wall_thickness_change),
        ]

        for name, label, vmin, vmax, val, step, color, callback in slider_specs:
            ax = fig.add_axes([slider_left, slider_y, slider_width, slider_height])
            slider = Slider(ax, label, vmin, vmax, valinit=val, valstep=step)
            self._format_slider(slider, color=color)
            slider.on_changed(callback)
            self._panel_sliders[name] = slider
            self._panel_axes_list.append(ax)
            slider_y -= slider_gap

        cubic = float(max(0.0, 1.0 - self.cfg.shot_friction_linear - self.cfg.shot_friction_quadratic))
        cubic_text = bg_ax.text(0.08, 0.18, f'Friction Cubic: {cubic:.3f}', color='#aaaaaa', fontsize=8.5, ha='left', va='top', transform=bg_ax.transAxes)
        self._panel_elements['cubic_text'] = cubic_text
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
            try:
                plt.close(self._panel_fig)
            except Exception:
                self._on_panel_close(None)
        else:
            self._panel_axes_list.clear()
            self._panel_sliders.clear()
            self._panel_elements.clear()
            self._panel_close_cid = None

    def _on_panel_close(self, _event):
        self._panel_axes_list.clear()
        self._panel_sliders.clear()
        self._panel_elements.clear()
        self._panel_updating = False
        self._panel_fig = None
        self._panel_close_cid = None
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
        move_slider = float(getattr(self.cfg, 'movement_speed_scale', 1.0))
        move_factor = getattr(self, '_movement_speed_factor', 1.0)
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
        values = {
            'lin': float(self.cfg.shot_friction_linear),
            'quad': float(self.cfg.shot_friction_quadratic),
            'min': float(self.cfg.shot_friction_min_scale),
            'sink': float(self.cfg.sink_prob_threshold),
            'boost': float(self.cfg.boost_hole_probability_increment),
            'move': float(self.cfg.movement_speed_scale),
            'time': float(self.cfg.shot_time_limit or 0.0),
            'wall': float(getattr(self.cfg, 'single_wall_thickness_factor', 1.0)),
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
        if 'boost' in sliders:
            sliders['boost'].valtext.set_text(f"{self.cfg.boost_hole_probability_increment:.3f}")
        if 'move' in sliders:
            sliders['move'].valtext.set_text(f"{self._movement_speed_factor:.2f}x")
        if 'time' in sliders:
            if self.cfg.shot_time_limit is None:
                sliders['time'].valtext.set_text('inf')
            else:
                sliders['time'].valtext.set_text(f"{self.cfg.shot_time_limit:.0f}")
        if 'wall' in sliders:
            sliders['wall'].valtext.set_text(f"{getattr(self.cfg, 'single_wall_thickness_factor', 1.0):.2f}")
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

    def _on_boost_increment_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        self.cfg.boost_hole_probability_increment = max(0.0, float(val))
        if self.cfg.boost_hole_probability_increment > 0.0 and not getattr(self.cfg, 'boost_hole_probability', False):
            self.cfg.boost_hole_probability = True
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
        slider = self._panel_sliders.get('time')
        if slider is None:
            return
        value = float(val)
        if value <= 1.0:
            self.cfg.shot_time_limit = None
        else:
            self.cfg.shot_time_limit = value
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()

    def _on_wall_thickness_change(self, val):
        if getattr(self, '_panel_updating', False):
            return
        thickness = float(np.clip(val, 0.05, 5.0))
        self.cfg.single_wall_thickness_factor = thickness
        if self.cfg.map_kind == 'single_wall':
            self.course.set_map(self.cfg.map_kind)
            self.course.update_exponents(self.cfg.dt, self.k2, self.c64)
            self.viz.set_course_patches(self.course.course_patches)
            if self._mode_allows_quantum():
                self._draw_idle_wave_preview()
            elif self._mode_allows_classical():
                self.viz.draw_frame(np.zeros((self.Ny, self.Nx), dtype=np.float32), plot_wave=False)
        self._refresh_slider_texts(draw=False)
        self._panel_draw_idle()
        self.viz.fig.canvas.draw_idle()

    def _reset(self, ball_only=False):
        start_x = int(
            round(max(8, min(self.Nx - 8, self.Nx * self.cfg.ball_start_x_frac))))
        self.ball_pos = np.array(
            [float(start_x), self.Ny / 2], dtype=np.float32)
        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])

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

        kvec = self._compute_kvec_from_swing(v_vec, closing_speed)
        if kvec is None:
            return
        self.viz.indicator_patch.set_visible(False)
        self.viz.update_title(self._title_text())
        self._shoot(kvec)

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
        dt_eff = old_dt

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
        self.course.update_exponents(dt_eff, self.k2, self.c64)

        try:
            sunk_prob = 0.0
            base_steps = self.base_steps_per_shot * (self.cfg.perf_steps_factor if self.perf_mode else 1.0)
            steps = int(max(1, round(base_steps)))
            shot_limit = getattr(self.cfg, 'shot_time_limit', None)
            if shot_limit is not None and shot_limit > 0.0:
                limit_steps = int(math.ceil(shot_limit / max(dt_eff, 1e-9)))
                max_steps = int(max(1, getattr(self.cfg, 'max_steps_per_shot', limit_steps)))
                steps = int(max(1, min(limit_steps, max_steps)))
            draw_every = self.perf_draw_every if self.perf_mode else self.base_draw_every
            sigma = self.cfg.perf_sigma0 if self.perf_mode else self.cfg.sigma0

            if self.cfg.flags.adaptive_draw:
                cw, ch = self.viz.fig.canvas.get_width_height()
                pixel = max(1, cw * ch)
                scale_adapt = pixel / (900 * 600)
                draw_every = max(
                    1, int(round(draw_every * scale_adapt * (30.0 / self.cfg.target_fps))))

            self.shot_in_progress = True
            self.game_over = False
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
            V_operator_cache = tmp_v_operator = k2_cache = tmp_k2 = None
            if simulate_wave and friction_mode:
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
                    s_pos = np.maximum(s, 0.0)
                    forward_gauss = np.exp(-(s_pos ** 2) /
                                           (2 * (sigma_forward ** 2))).astype(np.float32)
                    transition = 0.5 * (1.0 + np.tanh(s / trans_len)).astype(np.float32)
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
            else:
                Xg = Yg = None

            if simulate_classical:
                x0, y0 = self.ball_pos
                c_pos = np.array([float(x0), float(y0)], dtype=float)
                c_v = np.array(
                    [float(kvec_eff[0]), float(kvec_eff[1])], dtype=float)
                c_speed = np.linalg.norm(c_v) + 1e-9
                c_dtc = min(0.5, 0.6 / c_speed)
                c_t = 0.0
                class_xs = [c_pos[0]]
                class_ys = [c_pos[1]]
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

            for n in range(steps):
                if friction_mode:
                    speed_scale = self._friction_speed_scale(n, steps, friction_coeffs)
                    speed_scale = max(0.0, speed_scale)
                else:
                    speed_scale = 1.0

                if friction_mode and self._wavefront_active:
                    self._wavefront_kmag = kmag * speed_scale

                if simulate_wave:
                    if friction_mode:
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
                        psi = step_wave(psi, expV_half_eff, expK_eff, self.be.fft2, self.be.ifft2,
                                        inplace=self.cfg.flags.inplace_step)
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
                        step = min(c_dtc, t_target - c_t)
                        p2 = c_pos + c_v * step

                        # border reflect
                        if p2[0] < 0:
                            p2[0] = -p2[0]
                            c_v[0] = -c_v[0]
                        elif p2[0] > self.Nx:
                            p2[0] = 2 * self.Nx - p2[0]
                            c_v[0] = -c_v[0]
                        if p2[1] < 0:
                            p2[1] = -p2[1]
                            c_v[1] = -c_v[1]
                        elif p2[1] > self.Ny:
                            p2[1] = 2 * self.Ny - p2[1]
                            c_v[1] = -c_v[1]

                        # collide with solids
                        eps = 1e-3
                        for (rx1, ry1, rx2, ry2) in self.course.solid_rects:
                            if rx1 <= p2[0] <= rx2 and ry1 <= p2[1] <= ry2:
                                dl = abs(p2[0] - rx1)
                                dr = abs(rx2 - p2[0])
                                dtp = abs(p2[1] - ry1)
                                db = abs(ry2 - p2[1])
                                m = min(dl, dr, dtp, db)
                                if m == dl:
                                    p2[0] = rx1 - eps
                                    c_v[0] = -abs(c_v[0])
                                elif m == dr:
                                    p2[0] = rx2 + eps
                                    c_v[0] = abs(c_v[0])
                                elif m == dtp:
                                    p2[1] = ry1 - eps
                                    c_v[1] = -abs(c_v[1])
                                else:
                                    p2[1] = ry2 + eps
                                    c_v[1] = abs(c_v[1])

                        c_pos = p2
                        self.ball_pos[0] = float(c_pos[0])
                        self.ball_pos[1] = float(c_pos[1])
                        self.viz.set_ball_center(self.ball_pos[0], self.ball_pos[1])
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
                        xs = class_xs[::self.path_decim] if (
                            self.cfg.flags.path_decimation and self.path_decim > 1) else class_xs
                        ys = class_ys[::self.path_decim] if (
                            self.cfg.flags.path_decimation and self.path_decim > 1) else class_ys
                        self.viz.class_path_line.set_data(xs, ys)
                        if c_pos is not None:
                            self.viz.class_marker.center = (c_pos[0], c_pos[1])
                        self.viz.class_marker.set_visible(True)
                    if simulate_wave:
                        ex, ey = compute_expectation(
                            self.Xgrid, self.Ygrid, dens, xp, self.be.to_cpu)
                        self._last_ex, self._last_ey = ex, ey
                        if self.show_info and (n % self.cfg.overlay_every == 0):
                            ex2, ey2, a1, b1, ang = covariance_ellipse(
                                self.Xgrid,
                                self.Ygrid,
                                dens,
                                xp,
                                self.be.to_cpu,
                            )
                            self.viz.update_overlay_from_stats(
                                ex2, ey2, a1, b1, ang, show=True)
                        wave_xs.append(ex)
                        wave_ys.append(ey)
                        wx = wave_xs[::self.path_decim] if (
                            self.cfg.flags.path_decimation and self.path_decim > 1) else wave_xs
                        wy = wave_ys[::self.path_decim] if (
                            self.cfg.flags.path_decimation and self.path_decim > 1) else wave_ys
                        self.viz.wave_path_line.set_data(wx, wy)
                        self.viz.wave_path_line.set_visible(True)
                        if sink_mode == "prob_threshold":
                            p_in = float(self.be.to_cpu(
                                (dens[self.course.hole_mask]).sum()))
                            if p_in > float(self.cfg.sink_prob_threshold):
                                sunk = True
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

                if not self.cfg.flags.blitting:
                    plt.pause(0.0001)

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

        finally:
            # restore dt exponents
            if self.tracker:
                self._update_tracker_reference()
            self.course.update_exponents(old_dt, self.k2, self.c64)
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
        self.viz.show_messages(wave_hit=in_hole, ball_hit=False)
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
