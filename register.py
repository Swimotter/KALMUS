from kalmus.barcodes.BrightnessBarcode import BrightnessBarcode
from kalmus.barcodes.ColorBarcode import ColorBarcode
from kalmus.frames.BackgroundFrame import BackgroundFrame
from kalmus.frames.FocusFrame import FocusFrame
from kalmus.frames.ForgroundFrame import ForegroundFrame
from kalmus.frames.HighContrastFrame import HighContrastFrame
from kalmus.frames.LowContrastFrame import LowContrastFrame
from kalmus.frames.WholeFrame import WholeFrame
from kalmus.metrics.brightness_metrics.BrightBrightnessMetric import BrightBrightnessMetric
from kalmus.metrics.brightness_metrics.BrightestBrightnessMetric import BrightestBrightnessMetric
from kalmus.metrics.color_metrics.AverageColorMetric import AverageColorMetric
from kalmus.metrics.color_metrics.MedianColorMetric import MedianColorMetric
from kalmus.metrics.color_metrics.ModeColorMetric import ModeColorMetric
from kalmus.metrics.color_metrics.TopDominantColorMetric import TopDominantColorMetric
from kalmus.metrics.color_metrics.WeightedDominantColorMetric import WeightedDominantColorMetric


def register():
    _register_metrics()
    _register_frames()
    _register_barcodes()

def _register_metrics():
    _register_color_metrics()
    _register_brightness_metrics()

def _register_color_metrics():
    AverageColorMetric()
    MedianColorMetric()
    ModeColorMetric()
    TopDominantColorMetric()
    WeightedDominantColorMetric()

def _register_brightness_metrics():
    BrightBrightnessMetric()
    BrightestBrightnessMetric()

def _register_frames():
    WholeFrame()
    ForegroundFrame()
    BackgroundFrame()
    HighContrastFrame()
    LowContrastFrame()
    FocusFrame()

def _register_barcodes():
    ColorBarcode(None, None)
    BrightnessBarcode(None, None)
