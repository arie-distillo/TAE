import lancedb
import pyarrow as pa
from config import settings


class TacticalDatabase:
    """
    LanceDB wrapper for the TAE theater index.

    Schema stores one record per TILE (not per frame). Each record includes:
    - CLIP vector for semantic search
    - Tile provenance (parent frame path + pixel offsets)
    - Tile ground footprint (4 WGS84 corners via bilinear interpolation)
    - Sensor metadata (altitude, gimbal yaw, GSD)
    """

    def __init__(self):
        self.db = lancedb.connect(settings.VECTOR_DB_PATH)
        self.table_name = "theater_index"
        self.table = None

    def initialize_table(self, vector_dim=512):
        schema = pa.schema([
            # Embedding
            pa.field("vector",       pa.list_(pa.float32(), vector_dim)),

            # Tile provenance
            pa.field("image_path",   pa.string()),   # path to the TILE (temp or cached)
            pa.field("parent_path",  pa.string()),   # path to the original full frame
            pa.field("tile_x",       pa.int32()),    # tile top-left x in parent pixels
            pa.field("tile_y",       pa.int32()),    # tile top-left y in parent pixels
            pa.field("tile_w",       pa.int32()),    # tile width in pixels
            pa.field("tile_h",       pa.int32()),    # tile height in pixels

            # Frame-level GPS (drone position, not tile center)
            pa.field("lat",          pa.float64()),
            pa.field("lon",          pa.float64()),
            pa.field("alt_m",        pa.float32()),
            pa.field("gimbal_yaw",   pa.float32()),

            # Tile ground footprint (bilinear interpolation of frame footprint)
            pa.field("gsd_cm_px",    pa.float32()),
            pa.field("fp_nw_lat",    pa.float64()),
            pa.field("fp_nw_lon",    pa.float64()),
            pa.field("fp_ne_lat",    pa.float64()),
            pa.field("fp_ne_lon",    pa.float64()),
            pa.field("fp_se_lat",    pa.float64()),
            pa.field("fp_se_lon",    pa.float64()),
            pa.field("fp_sw_lat",    pa.float64()),
            pa.field("fp_sw_lon",    pa.float64()),
        ])

        if self.table_name not in self.db.table_names():
            self.table = self.db.create_table(self.table_name, schema=schema)
        else:
            self.table = self.db.open_table(self.table_name)

    def add_observation(
        self,
        vector,
        tile_path: str,
        telemetry: dict,
        tile_footprint: dict,
        tile_x: int,
        tile_y: int,
        tile_w: int,
        tile_h: int,
    ):
        """
        Writes one tile record to the index.

        Parameters
        ----------
        vector          : CLIP embedding (numpy array or list)
        tile_path       : path to the tile image file (used by VLM)
        telemetry       : dict from pose_metadata.json (frame-level)
        tile_footprint  : dict from SpatialEngine.compute_tile_footprint()
        tile_x, tile_y  : tile top-left corner in parent frame pixels
        tile_w, tile_h  : tile dimensions in pixels
        """
        self.table.add([{
            "vector":      vector.tolist() if hasattr(vector, 'tolist') else vector,
            "image_path":  str(tile_path),
            "parent_path": str(telemetry['full_path']),
            "tile_x":      int(tile_x),
            "tile_y":      int(tile_y),
            "tile_w":      int(tile_w),
            "tile_h":      int(tile_h),
            "lat":         float(telemetry['lat']),
            "lon":         float(telemetry['lon']),
            "alt_m":       float(telemetry['z']),
            "gimbal_yaw":  float(telemetry.get('gimbal_yaw', 0.0)),
            "gsd_cm_px":   float(tile_footprint['gsd_cm_px']),
            "fp_nw_lat":   float(tile_footprint['nw'][0]),
            "fp_nw_lon":   float(tile_footprint['nw'][1]),
            "fp_ne_lat":   float(tile_footprint['ne'][0]),
            "fp_ne_lon":   float(tile_footprint['ne'][1]),
            "fp_se_lat":   float(tile_footprint['se'][0]),
            "fp_se_lon":   float(tile_footprint['se'][1]),
            "fp_sw_lat":   float(tile_footprint['sw'][0]),
            "fp_sw_lon":   float(tile_footprint['sw'][1]),
        }])

    def add_observations_batch(self, rows: list[dict]):
        """Write all tile rows for one frame in a single LanceDB call."""
        if rows:
            self.table.add(rows)

    def semantic_search(self, query_vector, limit: int = 5, 
                        frames_to_return: int = 3) -> list:
        """
        Returns up to `frames_to_return` tiles, at most ONE tile per parent frame,
        chosen from the top-`limit` ANN results.
        """
        raw = self.table.search(query_vector).limit(limit).to_list()
        
        seen_parents: set[str] = set()
        diverse: list[dict] = []
        for row in raw:
            parent = row.get("parent_path", "")
            if parent not in seen_parents:
                seen_parents.add(parent)
                diverse.append(row)
            if len(diverse) >= frames_to_return:
                break
        return diverse

    def row_count(self) -> int:
        if self.table is None:
            return 0
        return self.table.count_rows()
