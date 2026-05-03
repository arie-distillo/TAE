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
            raw = f.read(65536)
            content = raw.decode('latin-1', errors='ignore')
            
            # XMP parsing (on the already-read buffer)
            rel_alt      = re.search(r'RelativeAltitude="([-+]?\d*\.\d+|\d+)"', content)
            gimbal_pitch = re.search(r'GimbalPitchDegree="([-+]?\d*\.\d+|\d+)"', content)
            gimbal_yaw   = re.search(r'GimbalYawDegree="([-+]?\d*\.\d+|\d+)"', content)
            gimbal_roll  = re.search(r'GimbalRollDegree="([-+]?\d*\.\d+|\d+)"', content)

            # EXIF parsing (rewind the same open file)
            f.seek(0)
            tags = exifread.process_file(f)

        # Everything below uses 'tags' and regex matches — file is closed, that's fine
        lat = self._to_decimal(tags.get('GPS GPSLatitude'), tags.get('GPS GPSLatitudeRef'))
        lon = self._to_decimal(tags.get('GPS GPSLongitude'), tags.get('GPS GPSLongitudeRef'))

        img_w = tags.get('EXIF ExifImageWidth') or tags.get('Image ImageWidth')
        img_h = tags.get('EXIF ExifImageLength') or tags.get('Image ImageLength')

        exif_alt = tags.get('GPS GPSAltitude')
        exif_alt_val = float(exif_alt.values[0].num) / float(exif_alt.values[0].den) if exif_alt else 0.0

        z_val = float(rel_alt.group(1)) if rel_alt and float(rel_alt.group(1)) != 0 else exif_alt_val

        return {
            "z":            z_val,
            "gimbal_pitch": float(gimbal_pitch.group(1)) if gimbal_pitch else -90.0,
            "gimbal_yaw":   float(gimbal_yaw.group(1))   if gimbal_yaw  else 0.0,
            "gimbal_roll":  float(gimbal_roll.group(1))  if gimbal_roll else 0.0,
            "lat":          lat,
            "lon":          lon,
            "img_w_px":     int(str(img_w)) if img_w else 4000,
            "img_h_px":     int(str(img_h)) if img_h else 3000,
            "full_path":    str(image_path)
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
    gen = TAESimGenerator(image_folder="../data/raw/sim_env_1", output_path="../data/raw/sim_env_1")
    gen.generate()