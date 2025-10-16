from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, Circle, Ellipse, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D


class Visuals:
    """
    Owns the Matplotlib figure and all artists. Provides blitting helpers,
    overlay updates, and simple setters used by the game loop.
    """

    def __init__(self, Nx, Ny, hole_center, hole_r, flags, cfg):
        self.Nx, self.Ny = Nx, Ny
        self.flags = flags
        self.cfg = cfg

        dpi = (cfg.low_dpi_value if flags.low_dpi else None)
        self.fig, self.ax = plt.subplots(figsize=(8.5, 5.5), dpi=dpi)
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(0, Nx)
        self.ax.set_ylim(0, Ny)
        self.ax.set_xticks([]); self.ax.set_yticks([])

        animated = bool(flags.blitting)
        interp = 'nearest' if flags.pixel_upscale else str(getattr(cfg, "vis_interpolation", "bilinear"))

        # ---------- main heatmap ----------
        self.wave_vmin = 0.0
        self.wave_vmax = 0.04
        self._wave_norm_range = max(self.wave_vmax - self.wave_vmin, 1e-6)
        cmap = plt.get_cmap('magma', 256)
        lut = (cmap(np.linspace(0.0, 1.0, 256)) * 255.0).astype(np.uint8)
        lut[:, 3] = 255  # enforce opaque alpha
        self._cmap_lut = lut
        self._cmap_lut_gpu = {}
        init_rgba = np.zeros((Ny, Nx, 4), dtype=np.uint8)
        init_rgba[..., 3] = 255
        self._frame_rgba = init_rgba
        self.im = self.ax.imshow(
            init_rgba,
            origin='lower', extent=[0, Nx, 0, Ny],
            interpolation=interp, animated=animated
        )
        self.im.set_zorder(0)
        self.im.set_alpha(0.95)
        if flags.pixel_upscale:
            try:
                self.im.set_resample(False)
            except Exception:
                pass

        # obstacles collection (set later via set_course_patches)
        self.pcoll = None

        # border
        self.border = Rectangle((1, 1), Nx - 2, Ny - 2, fill=False,
                                edgecolor='white', linewidth=3.0)
        self.ax.add_patch(self.border)
        self.border.set_zorder(2)

        # hole
        self.hole_patch = Circle(
            (float(hole_center[0]), float(hole_center[1])),
            radius=float(hole_r),
            facecolor=(1, 1, 1, 0.05),
            edgecolor='white', linewidth=1.2
        )
        self.ax.add_patch(self.hole_patch)
        self.hole_patch.set_zorder(3)

        # ball & indicator
        self.ball_patch = Circle(
            (hole_center[0], hole_center[1]),
            radius=float(cfg.ball_r),
            facecolor=(1.0, 1.0, 1.0, 1.0),
            edgecolor=(0.0, 0.0, 0.0, 1.0),
            linewidth=1.3,
            zorder=10,
        )
        self.ax.add_patch(self.ball_patch)
        if animated: self.ball_patch.set_animated(True)
        self.ball_patch.set_visible(True)

        self.indicator_patch = Circle((0, 0), radius=float(cfg.indicator_r),
                                      facecolor=(0.7, 0.0, 0.9, 0.9),
                                      edgecolor='black', linewidth=0.6, zorder=11)
        self.ax.add_patch(self.indicator_patch)
        self.indicator_patch.set_visible(False)
        if animated: self.indicator_patch.set_animated(True)

        # ---------- path overlays ----------
        self.class_path_line, = self.ax.plot([], [], color='white', lw=2.0, linestyle='--', label='Classical path')
        self.wave_path_line,  = self.ax.plot([], [], color='lightskyblue', lw=2.0, label=r'$\langle r \rangle$ path (I/OFF)')
        self.class_path_line.set_zorder(9)
        self.wave_path_line.set_zorder(8)
        if animated:
            self.class_path_line.set_animated(True)
            self.wave_path_line.set_animated(True)

        # markers
        self.class_marker, = self.ax.plot([], [], marker='o', markersize=6,
                                          markerfacecolor='white', markeredgecolor='black',
                                          linestyle='None')
        self.class_marker.set_visible(False)
        if animated: self.class_marker.set_animated(True)

        self.wave_end_marker, = self.ax.plot([], [], marker='o', markersize=6,
                                             markerfacecolor='lightskyblue', markeredgecolor='white',
                                             linestyle='None')
        self.wave_end_marker.set_visible(False)
        if animated: self.wave_end_marker.set_animated(True)

        # measurement marker (text + point)
        self.measure_point, = self.ax.plot([], [], marker=None, markersize=0,
                                           color='cyan', linestyle='None')
        if animated: self.measure_point.set_animated(True)
        self.measure_marker = self.ax.text(0, 0, r'$e^-$', color='cyan', fontsize=12,
                                           ha='left', va='bottom')
        self.measure_marker.set_visible(False)
        if animated: self.measure_marker.set_animated(True)

        # wave crosshair + sigma ellipses
        self.wave_cross_hx, = self.ax.plot([], [], color='cyan', lw=1.0)
        self.wave_cross_hy, = self.ax.plot([], [], color='cyan', lw=1.0)
        if animated:
            self.wave_cross_hx.set_animated(True)
            self.wave_cross_hy.set_animated(True)

        self.sigma1_ell = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                                  edgecolor='lightskyblue', linewidth=1.2, linestyle='--')
        self.sigma2_ell = Ellipse((0, 0), 1, 1, angle=0, fill=False,
                                  edgecolor='lightskyblue', linewidth=1.2, linestyle='--')
        self.sigma1_ell.set_visible(False); self.sigma2_ell.set_visible(False)
        self.ax.add_patch(self.sigma1_ell); self.ax.add_patch(self.sigma2_ell)
        if animated:
            self.sigma1_ell.set_animated(True)
            self.sigma2_ell.set_animated(True)

        # putter polygon (for tracker overlay)
        self.putter_patch = Polygon(np.zeros((4, 2), dtype=float),
                                    closed=True, fill=True,
                                    facecolor=(0.2, 0.8, 1.0, 0.20),
                                    edgecolor=(0.2, 0.8, 1.0, 0.95), linewidth=1.5)
        self.putter_patch.set_visible(False)
        self.ax.add_patch(self.putter_patch)
        if animated: self.putter_patch.set_animated(True)

        # labels
        self.mode_label = self.ax.text(
            6, Ny - 4, "", color='white', fontsize=12, fontweight='bold',
            va='top', ha='left', path_effects=[pe.withStroke(linewidth=2.5, foreground='black')]
        )
        if animated: self.mode_label.set_animated(True)

        self._wave_label = self.ax.text(
            0, 0, "", color='lightskyblue', fontsize=10, va='bottom', ha='left',
            path_effects=[pe.withStroke(linewidth=2, foreground='black')]
        )
        self._wave_label.set_visible(False)
        if animated: self._wave_label.set_animated(True)

        self.class_label = self.ax.text(
            Nx - 4, Ny - 4, "", color='white', fontsize=10, va='top', ha='right'
        )
        if animated: self.class_label.set_animated(True)

        self.hole_msg = self.ax.text(
            Nx / 2, Ny - 6, "Wavefunction hit - perfect shot!",
            color='purple', fontsize=18, fontweight='bold', ha='center', va='top',
            bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor='purple', boxstyle='round,pad=0.35')
        )
        self.hole_msg_ball = self.ax.text(
            Nx / 2, Ny - 6, "Hole in One!",
            color='cyan', fontsize=18, fontweight='bold', ha='center', va='top',
            bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor='cyan', boxstyle='round,pad=0.35')
        )
        if animated:
            self.hole_msg.set_animated(True)
            self.hole_msg_ball.set_animated(True)
        self.hole_msg.set_visible(False)
        self.hole_msg_ball.set_visible(False)

        # legend
        handles = [
            Line2D([0], [0], marker='o', markersize=8, markerfacecolor='white',
                   markeredgecolor='black', linestyle='None', label='Classical ball'),
            Line2D([0], [0], color='lightskyblue', linewidth=2, label=r'$\langle r \rangle$ path (I/OFF)'),
            Line2D([0], [0], color='lightskyblue', linewidth=1.2, label='1sigma/2sigma ellipse', linestyle='--'),
            Line2D([0], [0], marker=r'$e^-$', color='cyan', markersize=10, linestyle='None',
                   markeredgecolor='cyan', label='Measurement')
        ]
        self.leg = self.ax.legend(handles=handles, loc='lower left', framealpha=0.25, fontsize=9)
        self._wave_legend_text = None
        texts = self.leg.get_texts() if self.leg is not None else []
        if texts:
            try:
                self._wave_legend_text = texts[1]
            except IndexError:
                self._wave_legend_text = texts[0]
        self.leg.set_visible(True)

        # interference pattern axis (hidden until toggled)
        base_pos = self.ax.get_position()
        panel_width = 0.13
        panel_pad = 0.02
        panel_left = base_pos.x1 + panel_pad
        max_left = max(0.0, 0.98 - panel_width)
        panel_left = min(panel_left, max_left)
        self.pattern_ax = self.fig.add_axes([
            panel_left,
            base_pos.y0,
            panel_width,
            base_pos.height
        ])
        self.pattern_ax.set_facecolor('black')
        for spine in self.pattern_ax.spines.values():
            spine.set_color('white')
        self.pattern_ax.tick_params(colors='white', labelsize=8)
        self.pattern_ax.set_title('Interference', color='white', fontsize=9)
        self.pattern_ax.set_xlabel('Intensity', color='white', fontsize=8)
        self.pattern_ax.set_ylabel('y', color='white', fontsize=8)
        self.pattern_ax.yaxis.set_label_position('right')
        self.pattern_ax.yaxis.tick_right()
        self.pattern_ax.set_xlim(0.0, 1.0)
        self.pattern_ax.set_ylim(0.0, float(Ny))
        self.pattern_ax.grid(False)
        self.pattern_line, = self.pattern_ax.plot([], [], color='white', linewidth=1.5)
        self.pattern_ax.set_visible(False)
        self.pattern_line.set_visible(False)
        self._pattern_max = 1.0

        # blitting state
        self._blit_bg = None
        self._animated_artists = []
        self._flush_counter = 0
        self._flush_stride = 1
        if flags.blitting:
            self._init_blit()
        self._update_flush_stride()
        if flags.blitting:
            self.fig.canvas.mpl_connect('resize_event', self._handle_resize)

    # ------------------------------------------------------------------ patches / drawing

    def set_course_patches(self, patches):
        if self.pcoll is not None:
            try:
                self.pcoll.remove()
            except Exception:
                pass
        self.pcoll = PatchCollection(
            patches, facecolor=(1, 1, 1, 1), edgecolor=(0, 0, 0, 0.5), linewidths=1.3, zorder=3
        )
        self.ax.add_collection(self.pcoll)
        if self.flags.blitting:
            self._init_blit()
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def ensure_rgba_buffer(self, shape):
        return self._ensure_frame_buffer(shape)

    def get_cmap_lut(self, xp=None):
        if xp is None or xp is np:
            return self._cmap_lut
        key = id(xp)
        lut = self._cmap_lut_gpu.get(key)
        if lut is None:
            try:
                lut = xp.asarray(self._cmap_lut)
            except Exception:
                return self._cmap_lut
            self._cmap_lut_gpu[key] = lut
        return lut

    def _ensure_frame_buffer(self, shape):
        if self._frame_rgba is None or self._frame_rgba.shape[:2] != shape:
            self._frame_rgba = np.empty(shape + (4,), dtype=np.uint8)
            try:
                self.im.set_data(self._frame_rgba)
            except Exception:
                pass
        return self._frame_rgba

    def _coerce_frame_indices(self, frame_cpu):
        if isinstance(frame_cpu, np.ndarray) and frame_cpu.dtype == np.uint8:
            return frame_cpu
        arr = np.array(frame_cpu, dtype=np.float32, copy=True)
        np.subtract(arr, self.wave_vmin, out=arr)
        np.clip(arr, 0.0, self._wave_norm_range, out=arr)
        if self._wave_norm_range > 0:
            arr *= (255.0 / self._wave_norm_range)
        return arr.astype(np.uint8, copy=False)

    def _encode_rgba(self, frame_idx):
        shape = frame_idx.shape
        buf = self._ensure_frame_buffer(shape)
        self._cmap_lut.take(frame_idx, axis=0, out=buf)
        return buf

    def draw_frame(self, frame_cpu, plot_wave=True, pre_encoded=False):
        if pre_encoded:
            if not isinstance(frame_cpu, np.ndarray):
                frame_cpu = np.asarray(frame_cpu, dtype=np.uint8)
            shape = frame_cpu.shape[:2]
            rgba = self._ensure_frame_buffer(shape)
            if frame_cpu is not rgba:
                np.copyto(rgba, frame_cpu, casting='unsafe')
        else:
            frame_idx = self._coerce_frame_indices(frame_cpu)
            rgba = self._encode_rgba(frame_idx)
        self.im.set_visible(bool(plot_wave))
        if self.im.get_array() is not rgba:
            try:
                self.im.set_data(rgba)
            except Exception:
                pass
        else:
            self.im.stale = True
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def set_ball_center(self, x, y):
        self.ball_patch.center = (float(x), float(y))

    def set_ball_visible(self, visible):
        self.ball_patch.set_visible(bool(visible))

    def set_wave_visible(self, visible):
        self.im.set_visible(bool(visible))
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def set_wave_path_label(self, overlay_on):
        # keep label in sync with legend entry
        label = r'$\langle r \rangle$ path (I/ON)' if overlay_on else r'$\langle r \rangle$ path (I/OFF)'
        self.wave_path_line.set_label(label)
        if self._wave_legend_text is not None:
            self._wave_legend_text.set_text(label)
        if self.leg is not None:
            self.leg.stale = True
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def set_interference_visible(self, visible):
        if not hasattr(self, 'pattern_ax') or self.pattern_ax is None:
            return
        vis = bool(visible)
        self.pattern_ax.set_visible(vis)
        self.pattern_line.set_visible(vis)
        if not vis:
            self.pattern_line.set_data([], [])
        if self.flags.blitting:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

    def update_interference_pattern(self, intensity_profile):
        if not hasattr(self, 'pattern_ax') or self.pattern_ax is None:
            return
        intensity = np.asarray(intensity_profile, dtype=np.float32)
        if intensity.size == 0:
            self.pattern_line.set_data([], [])
            self.pattern_ax.set_xlim(0.0, max(self._pattern_max, 1.0))
        else:
            y = np.linspace(0.0, float(self.Ny), intensity.size, endpoint=False, dtype=np.float32)
            peak = float(np.max(intensity)) if np.isfinite(float(np.max(intensity))) else 0.0
            if peak <= 0.0:
                peak = 1.0
            if self.pattern_ax.get_visible():
                self._pattern_max = 0.85 * self._pattern_max + 0.15 * peak
            else:
                self._pattern_max = peak
            span = max(self._pattern_max, 1e-3)
            self.pattern_ax.set_xlim(0.0, span * 1.05)
            self.pattern_line.set_data(intensity, y)
        if self.flags.blitting:
            self.fig.canvas.draw()
        else:
            self.fig.canvas.draw_idle()

    def update_mode_label(self, text, color):
        self.mode_label.set_text(text or "")
        self.mode_label.set_color(color)
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def update_title(self, text, sunk=False):
        title = (text or "") + ("  HOLE OUT!" if sunk else "")
        try:
            self.fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    # ------------------------------------------------------------------ overlay helpers

    def clear_classical_overlay(self):
        self.class_path_line.set_data([], [])
        self.class_marker.set_visible(False)
        self.class_label.set_text("")

    def clear_wave_overlay(self, plot_wave=True):
        self.wave_path_line.set_data([], [])
        self.wave_path_line.set_visible(bool(plot_wave))
        self.wave_end_marker.set_visible(False)
        self.wave_cross_hx.set_visible(False)
        self.wave_cross_hy.set_visible(False)
        self.sigma1_ell.set_visible(False)
        self.sigma2_ell.set_visible(False)
        self._wave_label.set_visible(False)

    def set_info_visibility(self, show):
        if not show:
            self.sigma1_ell.set_visible(False)
            self.sigma2_ell.set_visible(False)
            self.wave_cross_hx.set_visible(False)
            self.wave_cross_hy.set_visible(False)
            self._wave_label.set_visible(False)
            self.measure_point.set_visible(False)
            self.measure_marker.set_visible(False)

    def update_overlay_from_stats(self, ex, ey, a1, b1, angle_deg, show):
        # ellipses
        self.sigma1_ell.center = (ex, ey); self.sigma1_ell.width = a1;   self.sigma1_ell.height = b1;   self.sigma1_ell.angle = angle_deg
        self.sigma2_ell.center = (ex, ey); self.sigma2_ell.width = 2*a1; self.sigma2_ell.height = 2*b1; self.sigma2_ell.angle = angle_deg
        self.sigma1_ell.set_visible(show); self.sigma2_ell.set_visible(show)
        # crosshair
        L = 8.0
        self.wave_cross_hx.set_data([ex - L, ex + L], [ey, ey])
        self.wave_cross_hy.set_data([ex, ex], [ey - L, ey + L])
        self.wave_cross_hx.set_visible(show); self.wave_cross_hy.set_visible(show)
        # label
        self._wave_label.set_position((ex + 5, ey + 5))
        self._wave_label.set_visible(show)

    def update_putter_overlay(self, center, length, thickness, angle_deg, visible):
        if not visible:
            self.putter_patch.set_visible(False)
            if self.flags.blitting:
                self._blit_draw()
            else:
                self.fig.canvas.draw_idle()
            return
        cx, cy = center
        half_l = 0.5 * float(length)
        half_t = 0.5 * float(thickness)
        rad = math.radians(float(angle_deg))
        cos_a = math.cos(rad); sin_a = math.sin(rad)
        corners = np.array([[-half_l, -half_t], [half_l, -half_t], [half_l,  half_t], [-half_l,  half_t]])
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        pts = (corners @ rot.T) + np.array([cx, cy])
        self.putter_patch.set_xy(pts)
        self.putter_patch.set_visible(True)
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    def set_measure_point(self, mx, my, visible):
        self.measure_point.set_data([mx], [my])
        self.measure_point.set_visible(bool(visible))
        self.measure_marker.set_position((mx, my))
        self.measure_marker.set_visible(bool(visible))

    def show_messages(self, wave_hit=False, ball_hit=False):
        self.hole_msg.set_visible(bool(wave_hit))
        self.hole_msg_ball.set_visible(bool(ball_hit))
        if self.flags.blitting:
            self._blit_draw()
        else:
            self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ blitting

    def _handle_resize(self, _event):
        if self.flags.blitting:
            self._init_blit()
        self._update_flush_stride()

    def _update_flush_stride(self):
        try:
            w, h = self.fig.canvas.get_width_height()
            pixel = max(1, int(w) * int(h))
        except Exception:
            pixel = 1
        base_area = 900 * 600
        if pixel <= base_area:
            stride = 1
        else:
            stride = int(max(2, min(8, round(math.sqrt(pixel / base_area)))))
        self._flush_stride = max(1, stride)
        if self._flush_counter >= self._flush_stride:
            self._flush_counter = 0

    def _init_blit(self):
        dynamic_artists = [
            self.im,
            self.mode_label,
            self.class_path_line, self.wave_path_line,
            self.class_marker, self.wave_end_marker,
            self.hole_msg, self.hole_msg_ball,
            self.measure_point, self.measure_marker, self.indicator_patch,
            self.sigma1_ell, self.sigma2_ell,
            self.wave_cross_hx, self.wave_cross_hy,
            self.ball_patch, self.putter_patch,
            self._wave_label, self.class_label
        ]
        self._animated_artists = [a for a in dynamic_artists if a is not None]
        self.fig.canvas.draw()
        self._blit_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _blit_draw(self):
        if self._blit_bg is None:
            self.fig.canvas.draw()
            self._blit_bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        else:
            self.fig.canvas.restore_region(self._blit_bg)
        for a in self._animated_artists:
            if a is None:
                continue
            try:
                if not a.get_visible():
                    continue
                self.ax.draw_artist(a)
            except Exception:
                pass
        self.fig.canvas.blit(self.ax.bbox)
        flush = False
        if self._flush_stride <= 1:
            flush = True
        else:
            self._flush_counter += 1
            if self._flush_counter >= self._flush_stride:
                self._flush_counter = 0
                flush = True
        if flush:
            try:
                self.fig.canvas.flush_events()
            except Exception:
                pass
