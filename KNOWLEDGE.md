Q: How to cut a video with data stream (DJI telemetry in protobuf format) to preserve all the strreams including data stream?
A: `ffmpeg -ss 00:00:10 -to 00:00:40 -i city1.mp4 -map 0:0 -map 0:1 -map 0:2 -c copy -copy_unknown -avoid_negative_ts make_zero city1-10s.mov`

