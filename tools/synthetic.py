import cv2
import numpy as np
import os
import piexif
from math import tan, radians

class TAESimulatorV3:
    def __init__(self, anchor_path, object_path, res_m_px=0.05):
        self.anchor = cv2.imread(anchor_path)
        self.obj = self._process_object(object_path)
        self.res = res_m_px 
        
    def _process_object(self, path):
        """Removes background for natural overlapping."""
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None: return None
        if img.shape[2] == 3:
            # Create alpha mask from background color (top-left pixel)
            bg = img[0,0]
            mask = cv2.inRange(img, bg-15, bg+15)
            alpha = cv2.bitwise_not(mask)
            img = cv2.merge([img[:,:,0], img[:,:,1], img[:,:,2], alpha])
        return img

    def _inject_metadata(self, file_path, lat, lon, alt, pitch=-90):
        """Injects DJI-compatible EXIF and XMP tags for TAE."""
        # 1. Standard EXIF (GPS)
        def to_deg(value, loc):
            if value < 0: loc_value = loc[0]
            elif value > 0: loc_value = loc[1]
            else: loc_value = ""
            abs_value = abs(value)
            deg = int(abs_value)
            t1 = (abs_value - deg) * 60
            min = int(t1)
            sec = round((t1 - min) * 60, 5)
            return ((deg, 1), (min, 1), (int(sec * 100), 100)), loc_value

        lat_deg, lat_ref = to_deg(lat, ["S", "N"])
        lon_deg, lon_ref = to_deg(lon, ["W", "E"])

        exif_dict = {"GPS": {
            piexif.GPSIFD.GPSLatitudeRef: lat_ref,
            piexif.GPSIFD.GPSLatitude: lat_deg,
            piexif.GPSIFD.GPSLongitudeRef: lon_ref,
            piexif.GPSIFD.GPSLongitude: lon_deg,
            piexif.GPSIFD.GPSAltitude: (int(alt * 100), 100)
        }}
        exif_bytes = piexif.dump(exif_dict)

        # 2. XMP Injection (For RelativeAltitude/GimbalPitch)
        # This matches the pattern TAE looks for in DJI/Drone logs.
        xmp_data = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
        <x:xmpmeta xmlns:x='adobe:ns:meta/'>
            <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
                <rdf:Description rdf:about='' xmlns:drone-dji='http://www.dji.com/drone-dji/'>
                    <drone-dji:RelativeAltitude>{alt:.2f}</drone-dji:RelativeAltitude>
                    <drone-dji:GimbalPitchDegree>{pitch:.2f}</drone-dji:GimbalPitchDegree>
                </rdf:Description>
            </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end='w'?>""".encode('utf-8')

        # Attach to the file
        piexif.insert(exif_bytes, file_path)
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Insert XMP packet after the APP1 segment
        with open(file_path, 'wb') as f:
            f.write(content + xmp_data)

    def generate(self, config):
        os.makedirs(config['output_dir'], exist_ok=True)
        out_w, out_h = 4056, 3040 # High resolution target
        
        # Calculate dynamic FOV crop on anchor
        fov_w_px = int((2 * config['altitude'] * tan(radians(config['fov']/2))) / self.res)
        fov_h_px = int(fov_w_px * (out_h / out_w))

        for i in range(config['frames']):
            # 1. Calculate trajectory (Linear move example)
            alpha = i / config['frames']
            cy = int(config['start'][0] * (1-alpha) + config['end'][0] * alpha)
            cx = int(config['start'][1] * (1-alpha) + config['end'][1] * alpha)
            
            # 2. Crop & Resize
            y1, x1 = cy - fov_h_px//2, cx - fov_w_px//2
            crop = self.anchor[y1:y1+fov_h_px, x1:x1+fov_w_px].copy()
            

            # 3. Object Injection (Fixed Relative Mapping)
            if self.obj is not None and config['n1'] <= i <= config['n2']:
                oy, ox = config['obj_world_pos']
                
                # Calculate where the object sits RELATIVE to this frame's top-left
                rel_y, rel_x = oy - y1, ox - x1
                
                # Scale object
                obj_scaled = cv2.resize(self.obj, (0,0), fx=config['obj_scale'], fy=config['obj_scale'])
                h, w = obj_scaled.shape[:2]
                
                # Only draw if the object is actually inside the current crop boundaries
                if (0 <= rel_x < crop.shape[1] - w) and (0 <= rel_y < crop.shape[0] - h):
                    # Standard Alpha Blending (using the 4th channel from _process_object)
                    alpha_s = obj_scaled[:, :, 3] / 255.0
                    alpha_l = 1.0 - alpha_s
                    
                    for c in range(0, 3):
                        crop[rel_y:rel_y+h, rel_x:rel_x+w, c] = (
                            alpha_s * obj_scaled[:, :, c] + 
                            alpha_l * crop[rel_y:rel_y+h, rel_x:rel_x+w, c]
                        )

            # 4. Save and Inject Metadata
            frame_res = cv2.resize(crop, (out_w, out_h))
            path = os.path.join(config['output_dir'], f"FRAME_{i:04d}.jpg")
            cv2.imwrite(path, frame_res)
            
            # Sync metadata with frame (Lat/Lon/Alt)
            lat = 32.0968 + (cy * 0.000001)
            lon = 34.8156 + (cx * 0.000001)
            self._inject_metadata(path, lat, lon, config['altitude'])

# --- Execution ---
sim = TAESimulatorV3('sea.jpg', 'boat1.png')
sim.generate({
    'frames': 50, 'n1': 25, 'n2': 30, # Temporal visibility
    'altitude': 100.0, 'fov': 84,     # Realistic altitude
    'start': (2000, 2000), 'end': (3000, 3000),
    'obj_world_pos': (2000, 2000), 'obj_scale': 0.5,
    'output_dir': '../data/raw/tae_test_dataset'
})