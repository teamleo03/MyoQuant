"""Import facade; Settings now uses physical units and segment returns instance IDs."""
from .config import Settings
from .models import Candidate, DiameterMeasurement, DiameterSummary
from .image_io import read_image, write_image, resize_for_analysis
from .segmentation_engine import segment
from .diameter_engine import measure_diameters, recompute_summaries
from .results_io import make_overlay, save_results
