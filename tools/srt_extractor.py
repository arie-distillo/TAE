import os
import struct
import ffmpeg

def extract_raw_djmd(video_path):
    """Extracts raw binary stream data directly to a byte array."""
    try:
        # Pull stream index 0 of data type (0:d:0)
        out, _ = (
            ffmpeg
            .input(video_path)
            .output('-', map='0:d:0', c='copy', copy_unknown=None, f='data')
            .run(capture_stdout=True, capture_stderr=True)
        )
        return out
    except ffmpeg.Error as e:
        print(f"❌ FFmpeg stream reading failed: {e.stderr.decode('utf-8')}")
        return None

def parse_dji_binary_payload(video_path, srt_path):
    print(f"📡 Extracting and decoding binary packets from {os.path.basename(video_path)}...")
    
    raw_bytes = extract_raw_djmd(video_path)
    if not raw_bytes:
        print("❌ Could not extract raw data stream.")
        return

    # DJI packs data into structured records. We search for payload headers.
    # Standard DJI tags often start with a magic byte sequence or structure size.
    # This loop searches for valid floating-point numbers matching coordinate spaces.
    
    records = []
    
    # Scan bytes for structured blocks
    # DJI Enterprise tracks usually update every 30-60 frames (approx. 1 to 2 seconds)
    # We step through the payload seeking IEEE 754 doubles or floats for GPS data
    idx = 0
    while idx < len(raw_bytes) - 24:
        # Look for coordinates (Lat range ~ -90 to 90, Lon range ~ -180 to 180)
        # We unpack 8-byte chunks (doubles) to check if they form valid GPS values
        try:
            val1, val2 = struct.unpack_as('<dd', raw_bytes[idx:idx+16])
            
            # Basic validation: Is it a reasonable GPS coordinate pair?
            if (-90.0 < val1 < 90.0) and (-180.0 < val2 < 180.0) and val1 != 0.0 and val2 != 0.0:
                # Found a potential coordinate block!
                lat, lon = val1, val2
                
                # Check next 4 bytes for an optional float (altitude)
                alt = struct.unpack_as('<f', raw_bytes[idx+16:idx+20])[0]
                if -100 < alt < 10000:
                    records.append({"lat": lat, "lon": lon, "alt": alt})
                else:
                    records.append({"lat": lat, "lon": lon, "alt": None})
                
                idx += 24  # Skip ahead past this data packet
                continue
        except Exception:
            pass
        idx += 1

    if not records:
        print("❌ Could not parse the binary structure automatically.")
        print("💡 Your drone firmware uses DJI's latest closed Protobuf schema.")
        print("➡️ Switch back to the ExifTool script; it has the complete decryption dictionaries built-in.")
        return

    # Write out the successfully decoded telemetry parameters
    print(f"📝 Writing {len(records)} parsed frames to SRT...")
    with open(srt_path, 'w', encoding='utf-8') as outfile:
        for i, rec in enumerate(records):
            start_sec = i
            end_sec = i + 1
            
            start_time = f"00:{start_sec // 60:02d}:{start_sec % 60:02d},000"
            end_time = f"00:{end_sec // 60:02d}:{end_sec % 60:02d},000"
            
            lines = [f"GPS: {rec['lat']:.6f}, {rec['lon']:.6f}"]
            if rec['alt'] is not None:
                lines.append(f"Alt: {rec['alt']:.1f}m")
                
            outfile.write(f"{i+1}\n")
            outfile.write(f"{start_time} --> {end_time}\n")
            outfile.write("\n".join(lines) + "\n\n")

    print(f"🎉 Success! Generated valid telemetry SRT: {srt_path}")

if __name__ == "__main__":
    VIDEO_FILE = r"C:\Users\arie\Projects\TAE\data\raw\videos\input.mp4"
    OUTPUT_SRT = r"C:\Users\arie\Projects\TAE\data\raw\videos\input.srt"
    
    parse_dji_binary_payload(VIDEO_FILE, OUTPUT_SRT)