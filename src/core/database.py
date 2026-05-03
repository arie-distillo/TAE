import lancedb
import pyarrow as pa
from config import settings

class TacticalDatabase:
    def __init__(self):
        self.db = lancedb.connect(settings.DB_PATH)
        self.table_name = "theater_index"
        self.table = None

    def initialize_table(self, vector_dim=512):
        schema = pa.schema([
            pa.field("vector",      pa.list_(pa.float32(), vector_dim)),
            pa.field("image_path",  pa.string()),
            pa.field("lat",         pa.float64()),
            pa.field("lon",         pa.float64()),
            pa.field("alt_m",       pa.float32()),
            pa.field("gimbal_yaw",  pa.float32()),
            pa.field("gsd_cm_px",   pa.float32()),
            pa.field("fp_nw_lat",   pa.float64()),
            pa.field("fp_nw_lon",   pa.float64()),
            pa.field("fp_ne_lat",   pa.float64()),
            pa.field("fp_ne_lon",   pa.float64()),
            pa.field("fp_se_lat",   pa.float64()),
            pa.field("fp_se_lon",   pa.float64()),
            pa.field("fp_sw_lat",   pa.float64()),
            pa.field("fp_sw_lon",   pa.float64()),
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
            return

        self.table = self.db.open_table(self.table_name)

    def add_observation(self, vector, telemetry: dict, footprint: dict):
        self.table.add([{
            "vector":     vector.tolist() if hasattr(vector, 'tolist') else vector,
            "image_path": str(telemetry['full_path']),
            "lat":        float(telemetry['lat']),
            "lon":        float(telemetry['lon']),
            "alt_m":      float(telemetry['z']),
            "gimbal_yaw": float(telemetry.get('gimbal_yaw', 0.0)),
            "gsd_cm_px":  float(footprint['gsd_cm_px']),
            "fp_nw_lat":  float(footprint['nw'][0]),
            "fp_nw_lon":  float(footprint['nw'][1]),
            "fp_ne_lat":  float(footprint['ne'][0]),
            "fp_ne_lon":  float(footprint['ne'][1]),
            "fp_se_lat":  float(footprint['se'][0]),
            "fp_se_lon":  float(footprint['se'][1]),
            "fp_sw_lat":  float(footprint['sw'][0]),
            "fp_sw_lon":  float(footprint['sw'][1]),
        }])

    def semantic_search(self, query_vector, limit=5):
        return self.table.search(query_vector).limit(limit).to_list()