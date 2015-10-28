from base import OutlierDataset, iter_sampled_outliers

from simple import load_square, load_square_noise, load_spiral, load_sine_noise
from simple import load_ring_line_square

from .uci_one_class import load_breast_benign, load_heart_healthy
from .uci_one_class import load_diabetes_absent, load_arrhythmia_normal
from .uci_one_class import load_hepatitis_normal, load_colon_normal