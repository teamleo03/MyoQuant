"""Group means use equal weight per eligible five-point candidate, never outcome matching."""
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import statistics
from .results_io import write_csv

GROUPS = ('CCM','CP6','CP12','CP25','CON')


def load_references(path):
    """Read per-myotube means in micrometres without changing reference values."""
    rows=[]
    seen=set()
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        reader=csv.DictReader(handle)
        required={'group','field_id','reference_myotube_id','mean_um'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Reference CSV requires '+', '.join(sorted(required)))
        for row in reader:
            row['group']='CON' if row['group']=='Control' else row['group']
            row['field_id']=int(row['field_id'])
            row['mean_um']=float(row['mean_um'])
            key=(row['group'],row['field_id'],row['reference_myotube_id'])
            if key in seen or not math.isfinite(row['mean_um']) or row['mean_um']<=0:
                raise ValueError(f'Invalid or duplicate reference: {key}')
            seen.add(key)
            rows.append(row)
    return rows


def image_statistics(record):
    settings=record.settings_used
    candidates={c.id:c.row() for c in record.candidates}
    eligible=[s for s in record.summaries if eligible_for_group(candidates[s.myotube_id],s.row(),
               settings.min_area,settings.min_aspect_ratio,settings.min_eccentricity)]
    group,field=image_identity(record.path.name)
    return dict(group=group,field_id=field,candidates_total=len(record.candidates),
                eligible_means_um=[s.mean_diameter_um for s in eligible],
                ready_means_um=[s.mean_diameter_um for s in eligible if s.status=='Ready'])


def write_comparison(output, image_rows, references, complete=False):
    """A group-level fit is NOT a claim of verified individual detections."""
    rows=summarize_groups(image_rows,references)
    for row in rows:
        row['batch_complete']=complete
        row['individual_accuracy_validated']=False
    write_csv(Path(output)/'group_comparison.csv',rows)
    return rows


def image_identity(filename):
    stem=Path(filename).stem.lower().strip()
    found=re.fullmatch(r'(cp12|cp25|cp6|con|cm|p12|p25|p6)\s*(\d{2,3})',stem)
    if not found:
        return None, None
    prefix, field=found.groups()
    return {'cm':'CCM','con':'CON'}.get(prefix,prefix.upper()),int(field)


def eligible_for_group(candidate, summary, min_area=900, min_aspect=3., min_eccentricity=.88):
    """Fixed geometry/measurement rules, identical for every group.

    Review-tagged strands are included and counted separately. This is an
    exploratory automated population, not a set of expert-confirmed myotubes.
    """
    value=summary.get('mean_diameter_um')
    return (candidate['status']!='Excluded' and float(candidate['area_px'])>=min_area
            and float(candidate['aspect_ratio'])>=min_aspect
            and float(candidate['eccentricity'])>=min_eccentricity
            and int(summary['measurement_count'])==5
            and value not in (None,'') and math.isfinite(float(value)) and float(value)>0)


def summarize_groups(image_rows, references):
    result=[]
    for group in GROUPS:
        ref=[r for r in references if ('CON' if r['group']=='Control' else r['group'])==group and r['mean_um'] is not None]
        images=[r for r in image_rows if r['group']==group]
        values=[v for r in images for v in r['eligible_means_um']]
        ready=[v for r in images for v in r['ready_means_um']]
        ref_mean=statistics.mean(r['mean_um'] for r in ref) if ref else None
        mean=statistics.mean(values) if values else None
        error=100*abs(mean-ref_mean)/ref_mean if mean is not None and ref_mean else None
        shared={r['field_id'] for r in ref}
        shared_values=[v for r in images if r['field_id'] in shared for v in r['eligible_means_um']]
        represented={r['field_id'] for r in images if r['field_id'] in shared}
        shared_refs=[r['mean_um'] for r in ref if r['field_id'] in represented]
        result.append(dict(group=group,image_count=len(images),images_with_eligible_candidates=sum(bool(r['eligible_means_um']) for r in images),
                           candidates_total=sum(r['candidates_total'] for r in images),eligible_candidates=len(values),
                           ready_candidates=len(ready),review_candidates=len(values)-len(ready),reference_myotubes=len(ref),
                           reference_fields=len(shared),reference_mean_um=ref_mean,detected_mean_um=mean,
                           detected_candidate_sd_um=statistics.stdev(values) if len(values)>1 else None,
                           ready_only_mean_um=statistics.mean(ready) if ready else None,
                           absolute_relative_error_percent=error,within_5_percent=error is not None and error<=5,
                           shared_field_count=len(represented),
                           shared_fields_detected_mean_um=statistics.mean(shared_values) if shared_values else None,
                           shared_fields_reference_mean_um=statistics.mean(shared_refs) if shared_refs else None))
    return result
