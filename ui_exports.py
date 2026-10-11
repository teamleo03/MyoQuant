from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np
from .results_io import save_results, write_csv


class ExportMixin:
    def save_batch(self):
        if self._analysis_running:
            return
        self._store_current()
        records = [r for r in self.records if r.analyzed]
        if not records:
            messagebox.showinfo('Nothing to save', 'Analyze at least one image first.')
            return
        folder = filedialog.askdirectory(title='Choose results destination')
        if not folder:
            return
        root = Path(folder) / ('MyoQuant_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        combined, summaries = [], []
        try:
            for i, r in enumerate(records, 1):
                save_results(root / f'{i:03d}_{r.path.stem}', r.image_bgr, r.score, r.raw_mask,
                             r.accepted_mask, r.candidates, r.measurements, r.summaries, r.selected_ids,
                             settings=r.settings_used, source_path=r.path)
                combined += [dict(m.row(), image_name=r.path.name, image_number=i) for m in r.measurements]
                summaries += [dict(s.row(), image_name=r.path.name, image_number=i) for s in r.summaries]
            write_csv(root / 'all_diameter_measurements.csv', combined)
            write_csv(root / 'all_myotube_summary.csv', summaries)
            self._save_chart(root / 'myotube_diameter_chart.png', summaries)
        except Exception as exc:
            messagebox.showerror('Save failed', f'{exc}\nPartial output: {root}')
            return
        messagebox.showinfo('Saved', str(root))

    def _save_chart(self, path, rows):
        from matplotlib.figure import Figure
        rows = [r for r in rows if r['mean_diameter_um'] is not None]
        if not rows:
            return
        fig = Figure(figsize=(max(10, min(32, len(rows)*.12)), 6))
        ax = fig.subplots()
        ax.bar(np.arange(len(rows)), [r['mean_diameter_um'] for r in rows],
               color=['#dc9236' if r['status'] == 'Review' else '#3f9864' for r in rows])
        ax.set_ylabel('Mean diameter (µm)')
        ax.set_title('Five-point means; orange requires review')
        ax.set_xticks(np.arange(len(rows)))
        ax.set_xticklabels([f"{r['image_name']}:{r['myotube_id']}" for r in rows], rotation=90, fontsize=6)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
