from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .config import DEFAULT_CONFIG, load_config, save_config, settings_from_values
from .image_io import read_image
from .pipeline import ImageRecord
from .ui_analysis import AnalysisMixin
from .ui_exports import ExportMixin
from .ui_viewer import ViewerMixin


class App(AnalysisMixin, ExportMixin, ViewerMixin, tk.Tk):
    def __init__(self, config_path=DEFAULT_CONFIG):
        settings = load_config(config_path)
        super().__init__()
        self.title('MyoQuant 2.1 development — Experimental boundary segmentation')
        self.geometry('1480x940')
        self.minsize(1080, 720)
        self.config_path = Path(config_path)
        self.records, self.current_index = [], None
        self.image_path = self.image_bgr = self.accepted_mask = None
        self.candidates, self.measurements, self.summaries = [], [], []
        self.selected_ids = set()
        self.active_settings = settings
        self._analysis_running = False
        self._events = queue.Queue()
        self._pending = 0
        self.tk_image = self.canvas_image_id = None
        self.display_mode = 'overlay'
        self._display_revision = 0
        self._base_image_cache = {}
        self._render_cache_key = None
        self.zoom = self.fit_scale = 1.0
        self.offset_x = self.offset_y = 0.0
        self.drag_mode = self.drag_start_canvas = self.drag_start_offset = None
        self.drag_measurement = self.drag_endpoint = None

        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill='x')
        for name, command in [('Add Images', self.add_images), ('Remove', self.remove_image),
                              ('Analyze Current', self.analyze), ('Analyze All', self.analyze_all),
                              ('Measure Selected', self.measure_selected), ('Original', self.show_original),
                              ('Overlay', self.show_overlay), ('Fit', self.fit_view),
                              ('−', lambda: self.zoom_at(.8)), ('+', lambda: self.zoom_at(1.25)),
                              ('Save Batch Results', self.save_batch)]:
            ttk.Button(toolbar, text=name, command=command).pack(side='left', padx=2)
        controls = ttk.LabelFrame(self, text='Settings (changes apply on the next analysis)', padding=6)
        controls.pack(fill='x', padx=8)
        self.vars = {}
        for i, (key, value) in enumerate(asdict(settings).items()):
            row, group = divmod(i, 4)
            ttk.Label(controls, text=key).grid(row=row, column=group*2, sticky='e', padx=4, pady=2)
            var = tk.StringVar(value=str(value).lower() if isinstance(value, bool) else str(value))
            ttk.Entry(controls, textvariable=var, width=10).grid(row=row, column=group*2+1, sticky='w')
            self.vars[key] = var
        configbar = ttk.Frame(self, padding=4)
        configbar.pack(fill='x')
        ttk.Button(configbar, text='Reload config.csv', command=self.reload_config).pack(side='left')
        ttk.Button(configbar, text='Save GUI to config.csv', command=self.save_gui_config).pack(side='left', padx=5)
        ttk.Label(configbar, text='Resolution: px/µm. Length: µm. Green: Ready; orange: Review; gray: Excluded.').pack(side='left')
        body = ttk.Panedwindow(self, orient='horizontal')
        body.pack(fill='both', expand=True, padx=8, pady=6)
        sidebar, viewer = ttk.Frame(body), ttk.Frame(body)
        body.add(sidebar, weight=0)
        body.add(viewer, weight=1)
        self.image_list = tk.Listbox(sidebar, width=30, exportselection=False)
        self.image_list.pack(fill='both', expand=True)
        self.image_list.bind('<<ListboxSelect>>', self._on_list_select)
        self.status = tk.StringVar(value='Load original images. Experimental detection requires visual review.')
        ttk.Label(viewer, textvariable=self.status, wraplength=1050).pack(fill='x')
        self.progress_text = tk.StringVar(value='Ready')
        ttk.Label(viewer, textvariable=self.progress_text).pack(fill='x')
        self.batch_progress = ttk.Progressbar(viewer, mode='determinate')
        self.batch_progress.pack(fill='x')
        self.canvas = tk.Canvas(viewer, background='#202020', highlightthickness=0, cursor='crosshair')
        self.canvas.pack(fill='both', expand=True)
        for event, handler in [('<Configure>', self._on_resize), ('<Button-1>', self._on_left_down),
                               ('<B1-Motion>', self._on_left_drag), ('<ButtonRelease-1>', self._on_left_up),
                               ('<Button-2>', self._on_pan_down), ('<B2-Motion>', self._on_pan_drag),
                               ('<ButtonRelease-2>', self._on_pan_up), ('<Button-3>', self._on_pan_down),
                               ('<B3-Motion>', self._on_pan_drag), ('<ButtonRelease-3>', self._on_pan_up),
                               ('<MouseWheel>', self._on_mousewheel)]:
            self.canvas.bind(event, handler)
        self.after(100, self._poll_analysis)

    def settings(self):
        return settings_from_values({key: var.get() for key, var in self.vars.items()})

    def reload_config(self):
        try:
            values = asdict(load_config(self.config_path))
            for key, value in values.items():
                self.vars[key].set(str(value).lower() if isinstance(value, bool) else str(value))
            self.status.set('Settings loaded. Analyze again to apply; existing results retain their saved calibration.')
        except Exception as exc:
            messagebox.showerror('Configuration error', str(exc))

    def save_gui_config(self):
        try:
            save_config(self.settings(), self.config_path)
            self.status.set(f'Saved {self.config_path.name}. Analyze again to apply changes.')
        except Exception as exc:
            messagebox.showerror('Configuration error', str(exc))

    def add_images(self):
        if self._analysis_running:
            return
        paths = filedialog.askopenfilenames(filetypes=[('Images', '*.png *.jpg *.jpeg *.tif *.tiff *.bmp')])
        known = {str(r.path.resolve()).casefold() for r in self.records}
        for name in paths:
            key = str(Path(name).resolve()).casefold()
            if key not in known:
                self.records.append(ImageRecord(Path(name)))
                known.add(key)
        self._refresh_image_list()
        if self.records and self.current_index is None:
            self._select_record(0)

    def remove_image(self):
        if self._analysis_running or self.current_index is None:
            return
        del self.records[self.current_index]
        self.current_index = None
        self.image_bgr = self.accepted_mask = self.image_path = None
        self.candidates, self.measurements, self.summaries, self.selected_ids = [], [], [], set()
        self.canvas.delete('all')
        self.canvas_image_id = self.tk_image = None
        if self.records:
            self._select_record(0)
        self._refresh_image_list()

    def _on_list_select(self, _event):
        selected = self.image_list.curselection()
        if selected and not self._analysis_running:
            self._store_current()
            self._select_record(int(selected[0]))

    def _select_record(self, index):
        record = self.records[index]
        try:
            if record.image_bgr is None:
                record.image_bgr = read_image(record.path)
        except Exception as exc:
            messagebox.showerror('Open failed', str(exc))
            return
        self.current_index = index
        self.image_path, self.image_bgr = record.path, record.image_bgr
        self.accepted_mask, self.candidates = record.accepted_mask, record.candidates
        self.measurements, self.summaries = record.measurements, record.summaries
        self.selected_ids = record.selected_ids
        self.active_settings = record.settings_used
        self.display_mode = 'overlay' if record.analyzed else 'original'
        self._invalidate_display()
        self.after_idle(self.fit_view)
        self._update_status()
        self.image_list.selection_clear(0, 'end')
        self.image_list.selection_set(index)

    def _store_current(self):
        if self.current_index is not None:
            r = self.records[self.current_index]
            r.measurements, r.summaries, r.selected_ids = self.measurements, self.summaries, self.selected_ids

    def _refresh_image_list(self):
        self.image_list.delete(0, 'end')
        for r in self.records:
            self.image_list.insert('end', f'{"✓" if r.analyzed else "·"} [{len(r.selected_ids)}] {r.path.name}')
        if self.current_index is not None:
            self.image_list.selection_set(self.current_index)

    def show_original(self):
        self.display_mode = 'original'
        self._invalidate_display()
        self._redraw()

    def show_overlay(self):
        self.display_mode = 'overlay'
        self._invalidate_display()
        self._redraw()


def main():
    App().mainloop()
