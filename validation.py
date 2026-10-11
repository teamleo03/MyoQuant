"""Only explicit spatial/annotated correspondence can establish per-cell accuracy."""
import math


def compare_matched(reference, predictions, matches, threshold_percent=5.0):
    """Keys: (image_id, cell_id). Matches are (image_id, reference_id, predicted_id).

    Missing/Review predictions do not pass. Do not create matches by sorting
    diameters: that would select on the outcome and bias the error estimate.
    """
    if not math.isfinite(threshold_percent) or threshold_percent < 0:
        raise ValueError('Invalid threshold')
    used_reference, used_prediction = set(), set()
    rows = []
    for image_id, ref_id, pred_id in matches:
        ref_key, pred_key = (image_id, ref_id), (image_id, pred_id)
        if ref_key in used_reference or pred_key in used_prediction:
            raise ValueError('Correspondence must be one-to-one')
        if ref_key not in reference or pred_key not in predictions:
            raise ValueError('Unknown correspondence key')
        used_reference.add(ref_key)
        used_prediction.add(pred_key)
        expected, prediction = reference[ref_key], predictions[pred_key]
        if expected is None or not math.isfinite(expected) or expected <= 0:
            raise ValueError('Reference diameter must be finite and > 0')
        actual = prediction['mean_diameter_um']
        error = abs(actual-expected)/expected*100 if actual is not None and math.isfinite(actual) and actual > 0 else None
        rows.append(dict(image_id=image_id,reference_myotube_id=ref_id,predicted_myotube_id=pred_id,
                         reference_mean_um=expected,predicted_mean_um=actual,absolute_relative_error_percent=error,
                         within_5_percent=error is not None and error <= threshold_percent,
                         passes=error is not None and error <= threshold_percent and prediction['status']=='Ready',
                         status=prediction['status']))
    for image_id, ref_id in sorted(set(reference)-used_reference):
        rows.append(dict(image_id=image_id,reference_myotube_id=ref_id,predicted_myotube_id=None,
                         reference_mean_um=reference[(image_id,ref_id)],predicted_mean_um=None,
                         absolute_relative_error_percent=None,within_5_percent=False,passes=False,status='Unmatched'))
    return rows
