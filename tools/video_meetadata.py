import exifread
import re
import cv2

def extract_drone_metadata(video_path):
    # 1. Capture the first frame to read EXIF/XMP
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return "Could not read video file."

    # Save first frame temporarily to read metadata
    temp_img = "first_frame_temp.jpg"
    cv2.imwrite(temp_img, frame)

    metadata = {}

    with open(temp_img, 'rb') as f:
        # 2. Extract standard EXIF (GPS Lat/Lon)
        tags = exifread.process_file(f)
        
        # GPS extraction helper
        def get_if_exist(data, key):
            if key in data:
                return data[key]
            return None

        metadata['lat'] = get_if_exist(tags, 'GPS GPSLatitude')
        metadata['lon'] = get_if_exist(tags, 'GPS GPSLongitude')
        
        # 3. Extract DJI-Specific XMP Metadata using Regex
        # This bypasses standard parsers to find tactical data 
        f.seek(0)
        content = f.read().decode('latin-1')
        
        # Search for Relative Altitude [cite: 127]
        alt_match = re.search(r'relativeAltitude="([-+]?\d+\.\d+)"', content)
        metadata['relative_altitude'] = alt_match.group(1) if alt_match else "Not Found"
        
        # Search for Gimbal Pitch [cite: 135]
        pitch_match = re.search(r'GimbalPitchDegree="([-+]?\d+\.\d+)"', content)
        metadata['gimbal_pitch'] = pitch_match.group(1) if pitch_match else "Not Found"

    return metadata

# Example Usage
video_file = "7654074-hd_1280_720_25fps.mp4"
results = extract_drone_metadata(video_file)
print(f"Mission Metadata: {results}")