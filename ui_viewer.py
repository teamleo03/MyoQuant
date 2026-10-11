from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageTk
from .results_io import make_overlay
from .diameter_engine import recompute_summaries

class ViewerMixin:
    def _overlay_image(self):
        if self.image_bgr is None:
            return None
        if self.accepted_mask is None:
            return self.image_bgr
        return make_overlay(self.image_bgr, self.accepted_mask, self.candidates,
                            self.measurements, selected_ids=self.selected_ids)

    def _invalidate_display(self):
        self._display_revision += 1
        self._base_image_cache.clear()
        self._render_cache_key = None

    def _current_image(self):
        if self.image_bgr is None:
            return None
        key = (self.display_mode, self._display_revision)
        cached = self._base_image_cache.get(key)
        if cached is not None:
            return cached
        image = self.image_bgr if self.display_mode == 'original' or self.accepted_mask is None else self._overlay_image()
        self._base_image_cache[key] = image
        return image

    def fit_view(self):
        if self.image_bgr is None:
            return
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        h, w = self.image_bgr.shape[:2]
        self.fit_scale = min(cw / w, ch / h)
        self.zoom = 1.0
        self.offset_x = (cw - w * self.fit_scale) / 2
        self.offset_y = (ch - h * self.fit_scale) / 2
        self._redraw()

    def zoom_at(self, factor, canvas_x=None, canvas_y=None):
        if self.image_bgr is None:
            return
        canvas_x = self.canvas.winfo_width() / 2 if canvas_x is None else canvas_x
        canvas_y = self.canvas.winfo_height() / 2 if canvas_y is None else canvas_y
        old_scale = self._view_scale()
        image_x = (canvas_x - self.offset_x) / old_scale
        image_y = (canvas_y - self.offset_y) / old_scale
        self.zoom = float(np.clip(self.zoom * factor, 0.1, 20.0))
        new_scale = self._view_scale()
        self.offset_x = canvas_x - image_x * new_scale
        self.offset_y = canvas_y - image_y * new_scale
        self._redraw()

    def _view_scale(self):
        return max(self.fit_scale * self.zoom, 1e-9)

    def _canvas_to_image(self, x, y):
        scale = self._view_scale()
        return (x - self.offset_x) / scale, (y - self.offset_y) / scale

    def _on_resize(self, _event):
        self._redraw()

    def _on_mousewheel(self, event):
        self.zoom_at(1.2 if event.delta > 0 else 1 / 1.2, event.x, event.y)

    def _wheel_linux(self, event, direction):
        self.zoom_at(1.2 if direction > 0 else 1 / 1.2, event.x, event.y)

    def _on_pan_down(self, event):
        self.drag_mode = 'pan'
        self.drag_start_canvas = (event.x, event.y)
        self.drag_start_offset = (self.offset_x, self.offset_y)
        self.canvas.configure(cursor='fleur')

    def _on_pan_drag(self, event):
        if self.drag_mode != 'pan':
            return
        self.offset_x = self.drag_start_offset[0] + event.x - self.drag_start_canvas[0]
        self.offset_y = self.drag_start_offset[1] + event.y - self.drag_start_canvas[1]
        if self.canvas_image_id is not None:
            self.canvas.coords(self.canvas_image_id, self.offset_x, self.offset_y)

    def _on_pan_up(self, _event):
        if self.drag_mode == 'pan':
            self.drag_mode = None
            self.canvas.configure(cursor='crosshair')

    def _find_endpoint(self, image_x, image_y):
        threshold_px = 10 / self._view_scale()
        best, best_dist, endpoint_no = None, float('inf'), None
        for measurement in self.measurements:
            if measurement.diameter_px is None:
                continue
            for number, (x, y) in enumerate(((measurement.endpoint1_x, measurement.endpoint1_y),
                                             (measurement.endpoint2_x, measurement.endpoint2_y)), start=1):
                distance = float(np.hypot(image_x - x, image_y - y))
                if distance < threshold_px and distance < best_dist:
                    best, best_dist, endpoint_no = measurement, distance, number
        self.drag_endpoint = endpoint_no
        return best

    def _on_left_down(self, event):
        if self.image_bgr is None or self._analysis_running:
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        endpoint = self._find_endpoint(image_x, image_y)
        if endpoint is not None:
            self.drag_mode = 'endpoint'; self.drag_measurement = endpoint
            self.canvas.configure(cursor='hand2'); return
        candidate = self._candidate_at(image_x, image_y)
        if candidate is not None and candidate.status != 'Excluded':
            if candidate.id in self.selected_ids:
                self.selected_ids.remove(candidate.id)
                self.measurements = [m for m in self.measurements if m.myotube_id != candidate.id]
            else:
                self.selected_ids.add(candidate.id)
            self.summaries = recompute_summaries(self.measurements, self.active_settings, self.candidates)
            self._store_current(); self._refresh_image_list(); self._update_status()
            self._invalidate_display(); self._redraw(); return
        self._on_pan_down(event)

    def _on_left_drag(self, event):
        if self.drag_mode == 'endpoint' and self.drag_measurement is not None:
            image_x, image_y = self._canvas_to_image(event.x, event.y)
            image_x = float(np.clip(image_x, 0, self.image_bgr.shape[1]-1))
            image_y = float(np.clip(image_y, 0, self.image_bgr.shape[0]-1))
            m = self.drag_measurement
            if self.drag_endpoint == 1:
                m.endpoint1_x, m.endpoint1_y = image_x, image_y
            else:
                m.endpoint2_x, m.endpoint2_y = image_x, image_y
            m.center_x = (m.endpoint1_x + m.endpoint2_x) / 2
            m.center_y = (m.endpoint1_y + m.endpoint2_y) / 2
            m.diameter_px = float(np.hypot(m.endpoint2_x - m.endpoint1_x, m.endpoint2_y - m.endpoint1_y))
            settings = self.active_settings
            m.diameter_um = m.diameter_px / settings.pixels_per_um
            m.method, m.status, m.reason = 'manual', 'Review', 'manual_endpoint_edit'
            self.summaries = recompute_summaries(self.measurements, settings, self.candidates)
            self._store_current(); self._update_status(); self._invalidate_display(); self._redraw()
        elif self.drag_mode == 'pan':
            self._on_pan_drag(event)

    def _on_left_up(self, event):
        if self.drag_mode == 'endpoint':
            self.drag_mode = None; self.drag_measurement = None; self.drag_endpoint = None
            self.canvas.configure(cursor='crosshair')
        elif self.drag_mode == 'pan':
            self._on_pan_up(event)

    def _candidate_at(self, image_x, image_y):
        if self.accepted_mask is None:
            return None
        x, y = int(round(image_x)), int(round(image_y))
        if x < 0 or y < 0 or y >= self.accepted_mask.shape[0] or x >= self.accepted_mask.shape[1]:
            return None
        labels = self.accepted_mask
        label = int(labels[y, x])
        if label == 0:
            return None
        for candidate in self.candidates:
            if candidate.label == label:
                return candidate
        return None

    def _update_status(self):
        if self.image_path is None:
            return
        if self.accepted_mask is None:
            self.status.set(f'{self.image_path.name}: not analyzed.')
        elif self.summaries:
            ready = sum(s.status == 'Ready' for s in self.summaries)
            review = sum(s.status == 'Review' for s in self.summaries)
            self.status.set(f'{self.image_path.name}: {len(self.candidates)} candidates; {len(self.selected_ids)} selected; {ready} Ready; {review} Review. Calibration retained from analysis.')
        else:
            self.status.set(f'{self.image_path.name}: {len(self.selected_ids)} selected. Click Measure Selected after changing selections.')

    def _redraw(self):
        image_bgr = self._current_image()
        if image_bgr is None:
            return
        scale = self._view_scale()
        cache_key = (self.display_mode, self._display_revision, round(scale, 6))
        if self._render_cache_key != cache_key or self.tk_image is None:
            image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            width, height = max(1, int(round(image.width*scale))), max(1, int(round(image.height*scale)))
            image = image.resize((width, height), Image.Resampling.BILINEAR)
            self.tk_image = ImageTk.PhotoImage(image)
            self._render_cache_key = cache_key
            if self.canvas_image_id is None:
                self.canvas_image_id = self.canvas.create_image(self.offset_x, self.offset_y, image=self.tk_image, anchor='nw')
            else:
                self.canvas.itemconfigure(self.canvas_image_id, image=self.tk_image)
        self.canvas.coords(self.canvas_image_id, self.offset_x, self.offset_y)

