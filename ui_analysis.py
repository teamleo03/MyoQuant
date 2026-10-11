import queue
import threading
from tkinter import messagebox
from .pipeline import analyze_record
from .diameter_engine import measure_diameters


class AnalysisMixin:
    def analyze(self):
        if self.current_index is not None:
            self._start_analysis([self.current_index])

    def analyze_all(self):
        self._start_analysis(list(range(len(self.records))))

    def _start_analysis(self, indices):
        if self._analysis_running or not indices:
            return
        try:
            settings = self.settings()
        except Exception as exc:
            messagebox.showerror('Settings error', str(exc))
            return
        self._store_current()
        jobs = [(i, self.records[i]) for i in indices]
        self._analysis_running = True
        self._pending = len(jobs)
        self._batch_total = len(jobs)
        self.batch_progress.configure(maximum=len(jobs), value=0)
        self.progress_text.set(f'Analyzing {len(jobs)} original image(s) ...')

        def worker():
            for index, record in jobs:
                try:
                    self._events.put((index, analyze_record(record, settings), None))
                except Exception as exc:
                    self._events.put((index, None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _poll_analysis(self):
        while True:
            try:
                index, record, error = self._events.get_nowait()
            except queue.Empty:
                break
            self._pending -= 1
            if error:
                messagebox.showerror(f'Analysis failed: {self.records[index].path.name}', error)
            else:
                self.records[index] = record
                if index == self.current_index:
                    self._select_record(index)
            self.batch_progress.configure(value=self._batch_total-self._pending)
            self.progress_text.set(f'Completed {self._batch_total-self._pending} / {self._batch_total}')
            self._refresh_image_list()
            if self._pending == 0:
                self._analysis_running = False
        self.after(100, self._poll_analysis)

    def measure_selected(self):
        if self._analysis_running or self.accepted_mask is None:
            return
        self.measurements, self.summaries = measure_diameters(
            self.accepted_mask, self.candidates, self.active_settings, self.selected_ids)
        self._store_current()
        self._update_status()
        self._invalidate_display()
        self._redraw()
