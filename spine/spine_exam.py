import numpy as np


class InstanceMeta:
    def __init__(self, 
                 instance_number: int,
                 rows: int,
                 cols: int,
                 position: np.ndarray,
                 orientation: np.ndarray,
                 normal: np.ndarray,
                 projection: np.ndarray,
                 pixel_spacing: np.ndarray,
                 spacing_between_slices: float,
                 slice_thickness: float
                 ):
        self.instance_number = instance_number
        self.rows = rows
        self.cols = cols
        self.position = position
        self.orientation = orientation
        self.normal = normal
        self.projection = projection
        self.pixel_spacing = pixel_spacing
        self.spacing_between_slices = spacing_between_slices
        self.slice_thickness = slice_thickness


class SpineSeries:
    def __init__(self,
                study_id:int,
                series_id:int,
                description: str,
                volume: np.ndarray,
                meta: list[InstanceMeta]
                ):
        self.series_id = study_id
        self.series_id = series_id
        self.description = description
        self.volume = volume
        self.meta = meta

    def get_instance(self, instance_number: int) -> InstanceMeta:
        return self.meta[instance_number]


class SpineStudy:
    def __init__(self, 
                 study_id:int,
                 series: list[SpineSeries]):
        self.study_id = study_id
        self.series = series
        self.series_id_to_series = {s.series_id: s for s in series}

    def get_series(self, series_id: int) -> SpineSeries:
        return self.series_id_to_series[series_id]


    