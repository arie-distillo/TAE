import re
from pathlib import Path
import exifread


def dump_xmp(image_path):
    with open(image_path, 'rb') as f:
        content = f.read(65536).decode('latin-1', errors='ignore')
    
    fields = [
        'RelativeAltitude', 'AbsoluteAltitude',
        'GimbalPitchDegree', 'GimbalRollDegree', 'GimbalYawDegree',
        'FlightYawDegree', 'FlightPitchDegree', 'FlightRollDegree',
        'GPSLatitude', 'GPSLongitude'
    ]
    for field in fields:
        match = re.search(rf'{field}="([^"]+)"', content)
        print(f"{field:25s}: {match.group(1) if match else '--- MISSING ---'}")

    # If GPSLatitude and GPSLongitude are present, print them together
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f)

    gps_keys = [k for k in tags if 'GPS' in k]
    for k in gps_keys:
        print(f"{k}: {tags[k]}")

if __name__ == "__main__":
    dump_xmp("../data/raw/sim_env_1/DJI_20221020150359_0031_W.JPG")