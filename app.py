from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import re
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from .detector import (
    Candidate,
    DiameterMeasurement,
    DiameterSummary,
    Settings,
    make_overlay,
    measure_diameters,
    read_image,
    recompute_summaries,
    resize_for_analysis,
    save_results,
    segment,
)


@dataclass
class ImageRecord:
    path: Path
    source_bgr: np.ndarray | None = None
    image_bgr: np.ndarray | None = None
    analysis_scale: float = 1.0
    score: np.ndarray | None = None
    raw_mask: np.ndarray | None = None
    accepted_mask: np.ndarray | None = None
    candidates: list[Candidate] = field(default_factory=list)
    measurements: list[DiameterMeasurement] = field(default_factory=list)
    summaries: list[DiameterSummary] = field(default_factory=list)
    selected_ids: set[int] = field(default_factory=set)
    component_labels: np.ndarray | None = None
    analyzed: bool = False


class App(tk.Tk):
    MAX_IMAGES = 20
    MAX_SELECTED = 7

    def __init__(self):
        super().__init__()
        self.title('MyoQuant Pro v2.0 — Connected-Strand Analysis / Auto-select 7')
        self.geometry('1500x920')
        self.minsize(1080, 720)

        self.records: list[ImageRecord] = []
        self.current_index: int | None = None

        self.image_path: Path | None = None
        self.image_bgr = None
        self.score = None
        self.raw_mask = None
        self.accepted_mask = None
        self.candidates = []
        self.measurements = []
        self.summaries = []
        self.selected_ids: set[int] = set()
        self.component_labels = None
        self.active_scale = 1.0
        self._analysis_running = False

        self.tk_image = None
        self.canvas_image_id = None
        self.display_mode = 'overlay'
        self._display_revision = 0
        self._base_image_cache = {}
        self._render_cache_key = None

        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.fit_scale = 1.0
        self.drag_mode = None
        self.drag_start_canvas = None
        self.drag_start_offset = None
        self.drag_measurement = None
        self.drag_endpoint = None

        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill='x')
        ttk.Button(toolbar, text='Add Images (max 20)', command=self.add_images).pack(side='left', padx=4)
        ttk.Button(toolbar, text='Remove Image', command=self.remove_image).pack(side='left', padx=4)
        self.analyze_current_button = ttk.Button(toolbar, text='Analyze Current', command=self.analyze)
        self.analyze_current_button.pack(side='left', padx=4)
        self.analyze_all_button = ttk.Button(toolbar, text='Analyze All', command=self.analyze_all)
        self.analyze_all_button.pack(side='left', padx=4)
        ttk.Button(toolbar, text='Measure Selected', command=self.measure_selected).pack(side='left', padx=4)
        ttk.Separator(toolbar, orient='vertical').pack(side='left', fill='y', padx=8)
        ttk.Button(toolbar, text='Original', command=self.show_original).pack(side='left', padx=3)
        ttk.Button(toolbar, text='Overlay', command=self.show_overlay).pack(side='left', padx=3)
        ttk.Button(toolbar, text='Fit', command=self.fit_view).pack(side='left', padx=3)
        ttk.Button(toolbar, text='−', width=3, command=lambda: self.zoom_at(0.8)).pack(side='left', padx=2)
        ttk.Button(toolbar, text='+', width=3, command=lambda: self.zoom_at(1.25)).pack(side='left', padx=2)

        ttk.Label(toolbar, text='Results name:').pack(side='right', padx=(8, 3))
        self.results_name = tk.StringVar(value='MyoQuant_Results')
        ttk.Entry(toolbar, textvariable=self.results_name, width=22).pack(side='right', padx=3)
        ttk.Button(toolbar, text='Save Batch Results', command=self.save_batch).pack(side='right', padx=4)

        controls = ttk.LabelFrame(self, text='Detection and calibration', padding=8)
        controls.pack(fill='x', padx=8)
        self.vars = {}
        fields = [
            ('Minimum area', 'min_area', 900),
            ('Minimum length', 'min_length', 85),
            ('Minimum aspect ratio', 'min_aspect_ratio', 3.0),
            ('Minimum eccentricity', 'min_eccentricity', 0.88),
            ('Threshold offset', 'local_offset', 0.018),
            ('Analysis scale', 'analysis_scale', 0.25),
            ('Calibration pixels', 'calibration_pixels', 558),
            ('Calibration µm', 'calibration_microns', 250),
        ]
        for i, (label, key, default) in enumerate(fields):
            ttk.Label(controls, text=label).grid(row=0, column=i * 2, sticky='e', padx=(8, 3))
            var = tk.StringVar(value=str(default))
            ttk.Entry(controls, textvariable=var, width=9).grid(row=0, column=i * 2 + 1, sticky='w')
            self.vars[key] = var

        body = ttk.Panedwindow(self, orient='horizontal')
        body.pack(fill='both', expand=True, padx=8, pady=(6, 8))

        sidebar = ttk.Frame(body, width=270)
        body.add(sidebar, weight=0)
        ttk.Label(sidebar, text='Batch images').pack(anchor='w', padx=4, pady=(4, 2))
        self.image_list = tk.Listbox(sidebar, exportselection=False, width=36)
        self.image_list.pack(fill='both', expand=True, padx=4, pady=4)
        self.image_list.bind('<<ListboxSelect>>', self._on_list_select)
        ttk.Label(
            sidebar,
            text='✓ analyzed   [7] selected\nRed cross = detected overlap; connected arms are measured separately.',
            justify='left',
        ).pack(anchor='w', padx=4, pady=4)

        viewer = ttk.Frame(body)
        body.add(viewer, weight=1)
        self.status = tk.StringVar(value='Add up to 20 images. Analyze All selects 7 strand candidates and records 5 local thickness measurements per candidate.')
        ttk.Label(viewer, textvariable=self.status, padding=(4, 4)).pack(fill='x')
        progress_row = ttk.Frame(viewer, padding=(4, 0, 4, 4))
        progress_row.pack(fill='x')
        self.progress_text = tk.StringVar(value='Batch progress: ready')
        ttk.Label(progress_row, textvariable=self.progress_text, width=28).pack(side='left')
        self.batch_progress = ttk.Progressbar(
            progress_row, orient='horizontal', mode='determinate', maximum=1, value=0,
        )
        self.batch_progress.pack(side='left', fill='x', expand=True, padx=(6, 0))
        self.canvas = tk.Canvas(viewer, background='#202020', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)

        self.canvas.bind('<Configure>', self._on_resize)
        self.canvas.bind('<Button-1>', self._on_left_down)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_up)
        self.canvas.bind('<Button-2>', self._on_pan_down)
        self.canvas.bind('<B2-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-2>', self._on_pan_up)
        self.canvas.bind('<Button-3>', self._on_pan_down)
        self.canvas.bind('<B3-Motion>', self._on_pan_drag)
        self.canvas.bind('<ButtonRelease-3>', self._on_pan_up)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        self.canvas.bind('<Button-4>', lambda e: self._wheel_linux(e, 1))
        self.canvas.bind('<Button-5>', lambda e: self._wheel_linux(e, -1))

    def settings(self) -> Settings:
        try:
            return Settings(
                min_area=int(float(self.vars['min_area'].get())),
                min_length=float(self.vars['min_length'].get()),
                min_aspect_ratio=float(self.vars['min_aspect_ratio'].get()),
                min_eccentricity=float(self.vars['min_eccentricity'].get()),
                local_offset=float(self.vars['local_offset'].get()),
                analysis_scale=float(self.vars['analysis_scale'].get()),
                measurements_per_myotube=5,
                calibration_pixels=float(self.vars['calibration_pixels'].get()),
                calibration_microns=float(self.vars['calibration_microns'].get()),
            )
        except ValueError as exc:
            raise ValueError('One or more settings are not valid numbers.') from exc

    def analysis_settings(self, scale_override: float | None = None) -> Settings:
        """Scale pixel-based thresholds and calibration to the working image."""
        settings = self.settings()
        if scale_override is not None:
            settings.analysis_scale = scale_override
        if not 0.10 <= settings.analysis_scale <= 1.0:
            raise ValueError('Analysis scale must be between 0.10 and 1.00.')
        scale = settings.analysis_scale
        settings.calibration_pixels *= scale
        settings.min_area = max(1, round(settings.min_area * scale * scale))
        settings.min_length *= scale
        settings.max_width *= scale
        settings.background_kernel = max(3, round(settings.background_kernel * scale))
        settings.local_block = max(3, round(settings.local_block * scale))
        settings.nuclear_hole_area = max(1, round(settings.nuclear_hole_area * scale * scale))
        settings.envelope_closing_radius = max(1, round(settings.envelope_closing_radius * scale))
        settings.overlap_cut_padding = max(1, round(settings.overlap_cut_padding * scale))
        settings.strand_bridge_gap *= scale
        return settings

    def add_images(self):
        paths = filedialog.askopenfilenames(filetypes=[
            ('Images', '*.png *.jpg *.jpeg *.tif *.tiff *.bmp'), ('All files', '*.*')
        ])
        if not paths:
            return
        existing = {str(r.path.resolve()).lower() for r in self.records}
        available = self.MAX_IMAGES - len(self.records)
        added = 0
        for path_text in paths:
            path = Path(path_text)
            key = str(path.resolve()).lower()
            if key in existing or added >= available:
                continue
            self.records.append(ImageRecord(path=path))
            existing.add(key)
            added += 1
        if len(paths) > available:
            messagebox.showinfo('20-image limit', f'Only {available} additional image(s) were added. The batch limit is 20.')
        self._refresh_image_list()
        if self.records and self.current_index is None:
            self._select_record(0)

    def remove_image(self):
        if self.current_index is None:
            return
        del self.records[self.current_index]
        if not self.records:
            self.current_index = None
            self._clear_current()
        else:
            self._select_record(min(self.current_index, len(self.records) - 1))
        self._refresh_image_list()

    def _on_list_select(self, _event):
        selection = self.image_list.curselection()
        if selection:
            self._select_record(int(selection[0]))

    def _select_record(self, index: int):
        self._store_current()
        self.current_index = index
        record = self.records[index]
        if record.source_bgr is None:
            try:
                record.source_bgr = read_image(record.path)
                record.image_bgr = record.source_bgr
            except Exception as exc:
                messagebox.showerror('Open failed', str(exc))
                return
        self.image_path = record.path
        self.image_bgr = record.image_bgr
        self.score = record.score
        self.raw_mask = record.raw_mask
        self.accepted_mask = record.accepted_mask
        self.candidates = record.candidates
        self.measurements = record.measurements
        self.summaries = record.summaries
        self.selected_ids = record.selected_ids
        self.component_labels = record.component_labels
        self.active_scale = record.analysis_scale
        self.display_mode = 'overlay' if record.analyzed else 'original'
        self._invalidate_display()
        self.after_idle(self.fit_view)
        self._update_status()
        self.image_list.selection_clear(0, 'end')
        self.image_list.selection_set(index)
        self.image_list.see(index)

    def _store_current(self):
        if self.current_index is None or self.current_index >= len(self.records):
            return
        record = self.records[self.current_index]
        record.image_bgr = self.image_bgr
        record.score = self.score
        record.raw_mask = self.raw_mask
        record.accepted_mask = self.accepted_mask
        record.candidates = self.candidates
        record.measurements = self.measurements
        record.summaries = self.summaries
        record.selected_ids = self.selected_ids
        record.component_labels = self.component_labels

    def _clear_current(self):
        self.image_path = None
        self.image_bgr = self.score = self.raw_mask = self.accepted_mask = None
        self.candidates = []
        self.measurements = []
        self.summaries = []
        self.selected_ids = set()
        self.component_labels = None
        self.canvas.delete('all')
        self.canvas_image_id = None
        self.tk_image = None
        self.status.set('Add up to 20 images.')

    def _refresh_image_list(self):
        selected = self.current_index
        self.image_list.delete(0, 'end')
        for i, record in enumerate(self.records):
            status = '✓' if record.analyzed else '·'
            count = len(record.selected_ids)
            self.image_list.insert('end', f'{status} [{count}] {record.path.name}')
        if selected is not None and selected < len(self.records):
            self.image_list.selection_set(selected)

    def analyze(self):
        if self.image_bgr is None:
            messagebox.showinfo('No image', 'Add and select an image first.')
            return
        if self._analysis_running:
            return
        try:
            base_settings = self.settings()
            analysis_settings = self.analysis_settings()
            if self.current_index is not None:
                record = self.records[self.current_index]
                source = record.source_bgr if record.source_bgr is not None else self.image_bgr
                record.source_bgr = source
            else:
                source = self.image_bgr
            record_index = self.current_index
            self._analysis_running = True
            self.analyze_current_button.configure(state='disabled')
            self.analyze_all_button.configure(state='disabled')
            self.batch_progress.configure(mode='indeterminate')
            self.batch_progress.start(12)
            self.progress_text.set('Analyzing current image…')
            self.status.set(f'Analyzing at {base_settings.analysis_scale}× scale. The window will remain responsive.')

            def worker():
                try:
                    image = resize_for_analysis(source, base_settings.analysis_scale)
                    score, raw_mask, accepted_mask, candidates = segment(image, analysis_settings)
                    ranked = sorted(candidates, key=lambda c: (c.score, c.length_px, c.area_px), reverse=True)
                    selected_ids = {c.id for c in ranked[:self.MAX_SELECTED]}
                    measurements, summaries = measure_diameters(
                        accepted_mask, candidates, analysis_settings, selected_ids=selected_ids
                    ) if selected_ids else ([], [])
                    result = (image, score, raw_mask, accepted_mask, candidates, selected_ids, measurements, summaries)
                    self.after(0, lambda value=result: self._finish_current_analysis(record_index, source, base_settings.analysis_scale, value, None))
                except Exception as exc:
                    self.after(0, lambda error=exc: self._finish_current_analysis(record_index, source, base_settings.analysis_scale, None, error))

            threading.Thread(target=worker, daemon=True).start()
        except Exception as exc:
            messagebox.showerror('Detection failed', str(exc))

    def _finish_current_analysis(self, record_index, source, analysis_scale, result, error):
        self.batch_progress.stop()
        self.batch_progress.configure(mode='determinate', maximum=1, value=1)
        self._analysis_running = False
        self.analyze_current_button.configure(state='normal')
        self.analyze_all_button.configure(state='normal')
        if error is not None:
            self.progress_text.set('Analysis failed')
            messagebox.showerror('Detection failed', str(error))
            return

        image, score, raw_mask, accepted_mask, candidates, selected_ids, measurements, summaries = result
        if record_index is not None and record_index < len(self.records):
            record = self.records[record_index]
            record.source_bgr = source
            record.image_bgr = image
            record.analysis_scale = analysis_scale
            record.score = score
            record.raw_mask = raw_mask
            record.accepted_mask = accepted_mask
            record.candidates = candidates
            record.selected_ids = selected_ids
            record.measurements = measurements
            record.summaries = summaries
            record.component_labels = measure_components(accepted_mask)
            record.analyzed = True

        if record_index == self.current_index:
            self.image_bgr = image
            self.score = score
            self.raw_mask = raw_mask
            self.accepted_mask = accepted_mask
            self.candidates = candidates
            self.selected_ids = selected_ids
            self.measurements = measurements
            self.summaries = summaries
            self.component_labels = measure_components(accepted_mask)
            self.active_scale = analysis_scale
            self.display_mode = 'overlay'
            self._invalidate_display()
            self._redraw()
        self._refresh_image_list()
        self.progress_text.set('Current image complete')
        self._update_status()

    def analyze_all(self):
        if not self.records:
            messagebox.showinfo('No images', 'Add images first.')
            return
        original_index = self.current_index or 0
        errors = []
        total = len(self.records)
        self.batch_progress.configure(maximum=total, value=0)
        self.progress_text.set(f'Batch progress: 0 / {total}')
        self.update_idletasks()
        for index, record in enumerate(self.records):
            scale_text = self.vars['analysis_scale'].get()
            self.status.set(f'Analyzing image {index + 1} of {len(self.records)} at {scale_text}× scale: {record.path.name}')
            self.update_idletasks()
            try:
                source = record.source_bgr if record.source_bgr is not None else read_image(record.path)
                record.source_bgr = source
                base_settings = self.settings()
                analysis_settings = self.analysis_settings()
                image = resize_for_analysis(source, base_settings.analysis_scale)
                score, raw_mask, accepted_mask, candidates = segment(image, analysis_settings)
                ranked = sorted(candidates, key=lambda c: (c.score, c.length_px, c.area_px), reverse=True)
                selected_ids = {c.id for c in ranked[:self.MAX_SELECTED]}
                measurements, summaries = measure_diameters(
                    accepted_mask, candidates, analysis_settings, selected_ids=selected_ids
                ) if selected_ids else ([], [])
                record.image_bgr = image
                record.analysis_scale = base_settings.analysis_scale
                record.score = score
                record.raw_mask = raw_mask
                record.accepted_mask = accepted_mask
                record.candidates = candidates
                record.selected_ids = selected_ids
                record.measurements = measurements
                record.summaries = summaries
                record.component_labels = measure_components(accepted_mask)
                record.analyzed = True
            except Exception as exc:
                errors.append(f'{record.path.name}: {exc}')
            finally:
                completed = index + 1
                self.batch_progress.configure(value=completed)
                self.progress_text.set(f'Batch progress: {completed} / {total} — {record.path.name}')
                self.update_idletasks()
        self._refresh_image_list()
        self._select_record(min(original_index, len(self.records) - 1))
        if errors:
            messagebox.showwarning('Batch completed with errors', '\n'.join(errors[:8]))
        else:
            self.status.set(f'Analyzed {len(self.records)} images. Red crosses mark overlaps; review selections and edit cyan endpoints as needed.')
        self.progress_text.set(f'Batch progress: complete — {total} / {total} images')

    def measure_selected(self):
        if self.accepted_mask is None:
            messagebox.showinfo('No detections', 'Analyze the image first.')
            return
        if not self.selected_ids:
            messagebox.showinfo('No selection', 'Select at least one myotube.')
            return
        self.measurements, self.summaries = measure_diameters(
            self.accepted_mask, self.candidates, self.analysis_settings(self.active_scale), selected_ids=self.selected_ids
        )
        self._store_current()
        self._update_status()
        self._invalidate_display()
        self._redraw()

    def show_original(self):
        if self.image_bgr is not None:
            self.display_mode = 'original'; self._invalidate_display(); self._redraw()

    def show_overlay(self):
        if self.image_bgr is not None:
            self.display_mode = 'overlay'; self._invalidate_display(); self._redraw()

    def save_batch(self):
        self._store_current()
        measured_records = [r for r in self.records if r.analyzed and r.measurements]
        if not measured_records:
            messagebox.showinfo('Nothing to save', 'Analyze at least one image first.')
            return
        parent = filedialog.askdirectory(title='Choose where to save the named results folder')
        if not parent:
            return
        safe_name = re.sub(r'[^A-Za-z0-9._ -]+', '_', self.results_name.get().strip()).strip(' .')
        if not safe_name:
            safe_name = 'MyoQuant_Results'
        root = Path(parent) / safe_name
        root.mkdir(parents=True, exist_ok=True)

        combined_measurements = []
        combined_summaries = []
        for image_number, record in enumerate(measured_records, start=1):
            image_folder = root / f'{image_number:02d}_{record.path.stem}'
            selected_candidates = [c for c in record.candidates if c.id in record.selected_ids]
            selected_mask = selected_mask_for(record)
            save_results(
                image_folder, record.image_bgr, record.score, record.raw_mask, selected_mask,
                selected_candidates, record.measurements, record.summaries,
                selected_ids=record.selected_ids,
            )
            for m in record.measurements:
                row = m.row(); row.update({'image_number': image_number, 'image_name': record.path.name})
                combined_measurements.append(row)
            for s in record.summaries:
                row = s.row(); row.update({'image_number': image_number, 'image_name': record.path.name})
                combined_summaries.append(row)

        write_csv(root / 'all_diameter_measurements.csv', combined_measurements)
        write_csv(root / 'all_myotube_summary.csv', combined_summaries)
        self._save_chart(root / 'myotube_diameter_chart.png', combined_summaries)
        messagebox.showinfo('Saved', f'Named batch results saved to:\n{root}')

    def _save_chart(self, path: Path, rows: list[dict]):
        import matplotlib.pyplot as plt
        labels = [f"{r['image_number']:02d}:{Path(r['image_name']).stem} | T{r['myotube_id']}" for r in rows]
        values = [float(r['mean_diameter_um']) for r in rows]
        image_numbers = [int(r['image_number']) for r in rows]
        unique_images = sorted(set(image_numbers))
        cmap = plt.get_cmap('tab20', max(len(unique_images), 1))
        color_lookup = {number: cmap(i) for i, number in enumerate(unique_images)}
        colors = [color_lookup[n] for n in image_numbers]
        width = max(11, min(28, 0.18 * len(rows) + 8))
        fig, ax = plt.subplots(figsize=(width, 7))
        ax.bar(np.arange(len(values)), values, color=colors)
        ax.set_ylabel('Mean diameter (µm)')
        ax.set_xlabel('Image and selected myotube')
        ax.set_title(self.results_name.get().strip() or 'MyoQuant Results')
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    def _selected_mask(self):
        if self.accepted_mask is None:
            return None
        temp = ImageRecord(path=self.image_path or Path('image'), accepted_mask=self.accepted_mask,
                           candidates=self.candidates, selected_ids=self.selected_ids,
                           component_labels=self.component_labels)
        return selected_mask_for(temp)

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
            for number, (x, y) in enumerate(((measurement.endpoint1_x, measurement.endpoint1_y),
                                             (measurement.endpoint2_x, measurement.endpoint2_y)), start=1):
                distance = float(np.hypot(image_x - x, image_y - y))
                if distance < threshold_px and distance < best_dist:
                    best, best_dist, endpoint_no = measurement, distance, number
        self.drag_endpoint = endpoint_no
        return best

    def _on_left_down(self, event):
        if self.image_bgr is None:
            return
        image_x, image_y = self._canvas_to_image(event.x, event.y)
        endpoint = self._find_endpoint(image_x, image_y)
        if endpoint is not None:
            self.drag_mode = 'endpoint'; self.drag_measurement = endpoint
            self.canvas.configure(cursor='hand2'); return
        candidate = self._candidate_at(image_x, image_y)
        if candidate is not None:
            if candidate.id in self.selected_ids:
                self.selected_ids.remove(candidate.id)
                self.measurements = [m for m in self.measurements if m.myotube_id != candidate.id]
            else:
                if len(self.selected_ids) >= self.MAX_SELECTED:
                    messagebox.showinfo('Replace a selection', 'Deselect one automatic myotube first, then select the replacement.')
                    return
                self.selected_ids.add(candidate.id)
            self.summaries = recompute_summaries(self.measurements, self.settings())
            self._store_current(); self._refresh_image_list(); self._update_status()
            self._invalidate_display(); self._redraw(); return
        self._on_pan_down(event)

    def _on_left_drag(self, event):
        if self.drag_mode == 'endpoint' and self.drag_measurement is not None:
            image_x, image_y = self._canvas_to_image(event.x, event.y)
            m = self.drag_measurement
            if self.drag_endpoint == 1:
                m.endpoint1_x, m.endpoint1_y = image_x, image_y
            else:
                m.endpoint2_x, m.endpoint2_y = image_x, image_y
            m.center_x = (m.endpoint1_x + m.endpoint2_x) / 2
            m.center_y = (m.endpoint1_y + m.endpoint2_y) / 2
            m.diameter_px = float(np.hypot(m.endpoint2_x - m.endpoint1_x, m.endpoint2_y - m.endpoint1_y))
            settings = self.analysis_settings(self.active_scale)
            m.diameter_um = m.diameter_px * settings.calibration_microns / settings.calibration_pixels
            self.summaries = recompute_summaries(self.measurements, settings)
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
        labels = self.component_labels if self.component_labels is not None else measure_components(self.accepted_mask)
        label = int(labels[y, x])
        if label == 0:
            threshold = 16 / self._view_scale()
            nearby = sorted(self.candidates, key=lambda c: np.hypot(c.centroid_x-image_x, c.centroid_y-image_y))
            return nearby[0] if nearby and np.hypot(nearby[0].centroid_x-image_x, nearby[0].centroid_y-image_y) <= threshold else None
        for candidate in self.candidates:
            cx = int(np.clip(round(candidate.centroid_x), 0, labels.shape[1]-1))
            cy = int(np.clip(round(candidate.centroid_y), 0, labels.shape[0]-1))
            if labels[cy, cx] == label:
                return candidate
        return None

    def _update_status(self):
        if self.image_path is None:
            return
        if self.accepted_mask is None:
            self.status.set(f'{self.image_path.name}: not analyzed.')
        elif self.summaries:
            mean_um = float(np.mean([s.mean_diameter_um for s in self.summaries]))
            overlap_count = len({c.overlap_group for c in self.candidates if c.overlap_detected})
            overlap_text = f'; {overlap_count} overlap junction(s) flagged' if overlap_count else ''
            self.status.set(f'{self.image_path.name}: selected {len(self.selected_ids)}/7; {len(self.measurements)} measurements. Mean {mean_um:.2f} µm{overlap_text}. Drag cyan endpoints to adjust.')
        else:
            self.status.set(f'{self.image_path.name}: selected {len(self.selected_ids)}/7. Click Measure Selected after changing selections.')

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


def measure_components(mask):
    from skimage import measure
    return measure.label(mask, connectivity=2)


def selected_mask_for(record: ImageRecord):
    if record.accepted_mask is None:
        return None
    labels = record.component_labels if record.component_labels is not None else measure_components(record.accepted_mask)
    result = np.zeros_like(record.accepted_mask, dtype=bool)
    for candidate in record.candidates:
        if candidate.id not in record.selected_ids:
            continue
        x = int(np.clip(round(candidate.centroid_x), 0, result.shape[1]-1))
        y = int(np.clip(round(candidate.centroid_y), 0, result.shape[0]-1))
        label = labels[y, x]
        if label:
            result |= labels == label
    return result


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text('', encoding='utf-8-sig')
        return
    preferred = ['image_number', 'image_name']
    fields = preferred + [key for key in rows[0].keys() if key not in preferred]
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    App().mainloop()
