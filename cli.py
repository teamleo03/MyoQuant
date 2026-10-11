"""Headless processing. Each image result retains its own immutable settings."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import DEFAULT_CONFIG, load_config, Settings


def process_one(job):
    path, output, settings_dict, compare = job
    import cv2
    cv2.setNumThreads(1)
    from .pipeline import ImageRecord, analyze_record
    from .results_io import save_results
    started = time.perf_counter()
    settings = Settings(**settings_dict)
    r = analyze_record(ImageRecord(Path(path)), settings)
    save_results(Path(output)/'revised', r.image_bgr, r.score, r.raw_mask, r.accepted_mask,
                 r.candidates, r.measurements, r.summaries, r.selected_ids,
                 settings=settings, source_path=path)
    metrics = dict(image_path=str(path), shape=list(r.image_bgr.shape), candidates=len(r.candidates),
                   ready=sum(s.status=='Ready' for s in r.summaries),
                   review=sum(s.status=='Review' for s in r.summaries),
                   excluded=sum(c.status=='Excluded' for c in r.candidates),
                   complete_five=sum(s.measurement_count==5 for s in r.summaries),
                   output=str(output))
    from .group_comparison import image_statistics
    metrics.update(image_statistics(r))
    if compare:
        from .legacy_models import Settings as LegacySettings
        from .legacy_segmentation import segment
        from .legacy_diameter import measure_diameters
        from .legacy_results import save_results as save_legacy
        baseline = LegacySettings(analysis_scale=1., calibration_pixels=settings.pixels_per_um,
                                  calibration_microns=1., min_area=settings.min_area,
                                  min_length=settings.min_length_um*settings.pixels_per_um)
        score, raw, accepted, candidates = segment(r.image_bgr, baseline)
        selected = {c.id for c in candidates}
        measurements, summaries = measure_diameters(accepted, candidates, baseline, selected)
        save_legacy(Path(output)/'baseline_native', r.image_bgr, score, raw, accepted,
                    candidates, measurements, summaries, selected)
        metrics['baseline_candidates'] = len(candidates)
        (Path(output)/'baseline_native'/'settings_used.json').write_text(json.dumps(asdict(baseline), indent=2), encoding='utf-8')
    metrics['elapsed_seconds'] = round(time.perf_counter()-started, 2)
    (Path(output)/'run.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


def main():
    parser = argparse.ArgumentParser(description='MyoQuant original-image batch analysis')
    parser.add_argument('--input-dir', type=Path)
    parser.add_argument('--images', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--reference-csv', type=Path, help='Per-myotube manual means in micrometres for group comparison')
    parser.add_argument('--compare', action='store_true', help='Also run v2 segmentation at native resolution with shared calibration and length setting')
    args = parser.parse_args()
    if not args.input_dir and not args.images:
        from .app import App
        App(args.config).mainloop()
        return
    if args.output is None:
        parser.error('--output is required for batch processing')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    files = args.images or sorted(p for p in args.input_dir.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff','.bmp'})
    if not files:
        parser.error('No images found')
    # Saved overlays and scores are not original biological data.
    generated_names = {'overlay_with_diameters.png','raw_mask.png','accepted_mask.png','stain_score.png',
                       'cytoplasm_score.png','boundary_evidence.png','boundary_barriers.png','myotube_diameter_chart.png'}
    if any(p.name in generated_names for p in files):
        parser.error('Input includes generated overlays/masks. Select original image files only.')
    settings = load_config(args.config)
    from .group_comparison import load_references, write_comparison
    references=load_references(args.reference_csv) if args.reference_csv else None
    args.output.mkdir(parents=True, exist_ok=False)
    from .config import save_config
    save_config(settings,args.output/'config_used.csv')
    import hashlib
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (args.output/'source_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    jobs = [(str(p), str(args.output/f'{i:03d}_{p.stem}'), asdict(settings), args.compare) for i,p in enumerate(files,1)]
    results, errors = [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        pending = {executor.submit(process_one, job): job[0] for job in jobs}
        for future in as_completed(pending):
            try:
                result = future.result()
                results.append(result)
                print(f"{len(results)+len(errors)}/{len(files)} {Path(result['image_path']).name}: {result['candidates']} candidates, {result['ready']} Ready, {result['review']} Review", flush=True)
            except Exception as exc:
                errors.append({'image_path':pending[future], 'error':str(exc)})
                print(f'ERROR {pending[future]}: {exc}', flush=True)
            complete=len(results)==len(files) and not errors
            (args.output/'batch_manifest.json').write_text(json.dumps({'expected_images':len(files),'complete':complete,'results':results,'errors':errors},indent=2),encoding='utf-8')
            if references is not None:
                rows=write_comparison(args.output,results,references,complete)
                if complete:
                    print(json.dumps(rows,indent=2),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
