import os
import json
import exifread
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

class TAESimGenerator:
    def __init__(self, image_folder, output_path):
        self.image_folder = Path(image_folder).absolute()
        self.output_path = Path(output_path).absolute()
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.metadata = {}

    def _extract_dji_data(self, image_path):
        with open(image_path, 'rb') as f:
            content = f.read(30000).decode('latin-1', errors='ignore')
            
            # Altitude checks: Relative > Absolute > EXIF fallback
            rel_alt = re.search(r'RelativeAltitude="([-+]?\d*\.\d+|\d+)"', content)
            abs_alt = re.search(r'AbsoluteAltitude="([-+]?\d*\.\d+|\d+)"', content)
            pitch = re.search(r'GimbalPitchDegree="([-+]?\d*\.\d+|\d+)"', content)
            yaw = re.search(r'FlightYawDegree="([-+]?\d*\.\d+|\d+)"', content)

            f.seek(0)
            tags = exifread.process_file(f)
            lat = self._to_decimal(tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef'))
            lon = self._to_decimal(tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef'))
            
            exif_alt = tags.get('GPS GPSAltitude')
            exif_alt_val = float(exif_alt.values[0].num) / float(exif_alt.values[0].den) if exif_alt else 0.0

            z_val = 0.0
            if rel_alt and float(rel_alt.group(1)) != 0:
                z_val = float(rel_alt.group(1))
            elif abs_alt and float(abs_alt.group(1)) != 0:
                z_val = float(abs_alt.group(1))
            else:
                z_val = exif_alt_val

            return {
                "z": z_val,
                "pitch": float(pitch.group(1)) if pitch else -90.0,
                "yaw": float(yaw.group(1)) if yaw else 0.0,
                "lat": lat,
                "lon": lon,
                "full_path": str(image_path) # Stores the absolute path
            }

    def _to_decimal(self, coords, ref):
        if not coords: return 0.0
        d = float(coords.values[0].num) / float(coords.values[0].den)
        m = float(coords.values[1].num) / float(coords.values[1].den)
        s = float(coords.values[2].num) / float(coords.values[2].den)
        val = d + (m / 60.0) + (s / 3600.0)
        return -val if ref and ref.values[0] in ['S', 'W'] else val

    def generate(self):
        # Process all JPEGs in the source folder
        image_list = sorted([f for f in os.listdir(self.image_folder) if f.lower().endswith(('.jpg', '.jpeg'))])
        
        for filename in image_list:
            full_path = self.image_folder / filename
            try:
                self.metadata[filename] = self._extract_dji_data(full_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        # Save the metadata index
        with open(self.output_path / "pose_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)


if __name__ == "__main__":
    # Point this to your folder of DJI images
    gen = TAESimGenerator(image_folder="data/raw/sim_env_1", output_path="data/raw/sim_env_1")
    gen.generate()