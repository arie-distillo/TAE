from dataclasses import dataclass

@dataclass
class TelemetryFrame:
    full_path:    str
    lat:          float
    lon:          float
    z:            float   # AGL altitude in meters (RelativeAltitude)
    gimbal_pitch: float   # degrees, -90 = nadir
    gimbal_yaw:   float   # degrees, clockwise from North
    gimbal_roll:  float   # degrees
    img_w_px:     int
    img_h_px:     int