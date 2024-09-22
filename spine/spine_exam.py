import numpy as np
import pandas as pd

from typing import Optional

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
                 slice_thickness: float,
                 # original_pixel_spacing: np.ndarray|None = None,
                 # original_rows: int|None = None,
                 # original_cols: int|None = None,
                 scale: float|None = None,
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

        # self.original_pixel_spacing = original_pixel_spacing
        # self.original_rows = original_rows
        # self.original_cols = original_cols
        # self.original_pixel_spacing = original_pixel_spacing
        self.scale = scale


class SpineSeries:
    def __init__(self,
                study_id:int,
                series_id:int,
                description: str,
                volume: np.ndarray,
                meta: list[InstanceMeta],
                scale: Optional[float] = None
                ):
        self.series_id = study_id
        self.series_id = series_id
        self.description = description
        self.volume = volume
        self.meta = meta
        self.scale = scale
        self.instance_id_map = {inst.instance_number : inst for inst in self.meta}


    def get_instance(self, instance_number: int) -> InstanceMeta:
        return self.instance_id_map[instance_number]
    
    def get_instances(self):
        return self.meta


class SpineStudy:
    def __init__(self, 
                 study_id:int,
                 series: list[SpineSeries],
                 series_descriptions: list[str]):
        self.study_id = study_id
        self.series = series
        self.series_id_to_series = {s.series_id: s for s in series}
        self.description_series_map = {}
        for s, s_description in zip(series,series_descriptions):
            self.description_series_map.setdefault(s_description, []).append(s)

    def get_axial_t2(self) -> Optional[SpineSeries]:
        try:
            return self.description_series_map["Axial T2"][0]
        except KeyError or IndexError:
            return None
    
    def get_saggital_t2_stir(self) -> Optional[SpineSeries]:
        try:
            return self.description_series_map["Sagittal T2/STIR"][0]
        except KeyError or IndexError:
            return None

    def get_series(self, series_id: int) -> SpineSeries:
        return self.series_id_to_series[series_id]
    
    def get_instances_scales(self):
        return pd.DataFrame([[self.study_id,
                              s.series_id, instance.instance_number, instance.scale] 
                              for s in self.series for instance in s.meta.values()], columns=["study_id", "series_id",  "instance_number", "scale"])
    
    def get_series_scales(self):
        return pd.DataFrame([[self.study_id,
                              s.series_id, s.scale] 
                              for s in self.series], columns=["study_id", "series_id", "scale"])


    