from config import settings

class SpatialEngine:
    def __init__(self, sensor_width=None, focal_length=None):
        # Fallback to config if nothing is passed specifically
        self.sensor_width = float(sensor_width or settings.SENSOR_WIDTH_MM)
        self.focal_length = float(focal_length or settings.FOCAL_LENGTH_MM)

    def get_projection_data(self, telemetry, img_width_px):
        # We use the 'z' we finally fixed in the JSON!
        alt = float(telemetry.get('z', 0.0))
        
        if alt > 0:
            # GSD Calculation: (Alt * SensorWidth) / (FocalLength * ImageWidth)
            gsd = (alt * self.sensor_width) / (self.focal_length * img_width_px)
        else:
            gsd = 0.05 # Default 5cm/px
            
        return {
            "gsd_m_px": gsd,
            "ground_width_m": gsd * img_width_px,
            "center_gps": (telemetry['lat'], telemetry['lon'])
        }