# Sample Video

The backend works with:
1. **Webcam** — set `VIDEO_SOURCE=0` in `.env`
2. **Video file** — set `VIDEO_SOURCE=path/to/video.mp4` in `.env`

## Download a free sample video

For testing without a webcam, download a free traffic video:

**Option 1 — Pexels (free stock footage)**
1. Visit https://www.pexels.com/search/videos/traffic%20intersection/
2. Choose any traffic/street video (HD recommended)
3. Click "Free Download"
4. Save as `sample_video/traffic.mp4`

**Option 2 — YouTube (free stock)**
1. Visit a free stock video e.g. https://www.youtube.com/watch?v=MNn9qKGgkUo
2. Download with `yt-dlp`:
   ```bash
   yt-dlp -f "best[height<=720]" -o sample_video/traffic.mp4 "https://www.youtube.com/watch?v=MNn9qKGgkUo"
   ```

**Option 3 — Synthetic mode (no video needed)**
If no video source is available, the backend auto-generates a synthetic demo scene with moving cars, trucks, pedestrians, bicycles, a bus, and a motorcycle. Just leave `VIDEO_SOURCE` pointing to a non-existent device and it'll fall back automatically.

Then update your `.env`:
```
VIDEO_SOURCE=sample_video/traffic.mp4
```

## Alternatively: stream from YouTube directly

```
VIDEO_SOURCE=https://www.youtube.com/watch?v=MNn9qKGgkUo
```

(Note: YouTube stream support requires `yt-dlp` installed and may need additional setup.)
