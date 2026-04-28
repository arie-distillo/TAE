import lancedb
import pyarrow as pa
import pandas as pd
from config import settings

class TacticalDatabase:
    def __init__(self):
        self.db = lancedb.connect(settings.DB_PATH)
        self.table_name = "theater_index"

    def initialize_table(self, vector_dim=512):
        # Explicitly define the schema to avoid "Null" type inference
        schema = pa.schema([
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("image_path", pa.string()),
            pa.field("lat", pa.float64()),
            pa.field("lon", pa.float64()),
            pa.field("footprint", pa.float64())
        ])
        
        # Create (or overwrite) the table with the fixed schema
        self.table = self.db.create_table(
            self.table_name, 
            schema=schema, 
            mode="overwrite"
        )

    def add_observation(self, vector, image_path, telemetry, footprint):
        # Ensure vector is a list of floats
        self.table.add([{
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector,
            "image_path": str(image_path),
            "lat": float(telemetry['lat']),
            "lon": float(telemetry['lon']),
            "footprint": float(footprint)
        }])

    def semantic_search(self, query_vector, limit=5):
        return self.table.search(query_vector).limit(limit).to_list()