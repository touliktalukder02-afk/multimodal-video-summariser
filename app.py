import os
import base64
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import cv2
from flask import Flask, render_template_string, request
from groq import Groq
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_UPLOAD_MB", "300")) * 1024 * 1024

TRANSCRIPTION_MODEL = os.environ.get("GROQ_TRANSCRIPTION_MODEL", "whisper-large-v3")
VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
SUMMARY_MODEL = os.environ.get("GROQ_SUMMARY_MODEL", "llama-3.3-70b-versatile")
MAX_FRAMES = max(1, min(int(os.environ.get("MAX_FRAMES", "12")), 12))
FRAME_INTERVAL_SECONDS = 30
AUDIO_CHUNK_SECONDS = 60
WORK_ROOT = Path(tempfile.gettempdir()) / "video_summariser"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

PAGE = r"""
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Multimodal Video Summariser</title>
<style>
body{font-family:Arial,sans-serif;background:#f4f6fa;color:#172033;margin:0;padding:24px}
main{max-width:900px;margin:24px auto;background:white;padding:28px;border-radius:14px;box-shadow:0 4px 24px #0001}
h1{margin-top:0}p{line-height:1.5}.note{background:#eff6ff;padding:12px;border-radius:8px}
input,button{font:inherit}input{display:block;margin:14px 0}button{border:0;border-radius:8px;background:#1d4ed8;color:white;padding:12px 18px;cursor:pointer}
.error{white-space:pre-wrap;background:#fff1f2;color:#9f1239;padding:14px;border-radius:8px}.result{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.55}
details{margin-top:18px}small{color:#596579}
</style></head><body><main>
<h1>Multimodal Video Summariser</h1>
<p>Upload a video to summarise its spoken audio and sampled visual frames using Groq.</p>
<p class="note"><small>Processing can take several minutes and uses your Groq API quota. Upload only videos you are permitted to process. Files and temporary frames are deleted after the request finishes.</small></p>
{% if error %}<div class="error">{{ error }}</div>{% endif %}
<form method="post" enctype="multipart/form-data">
<label for="video"><b>Select a video (MP4 recommended)</b></label>
<input id="video" name="video" type="file" accept="video/*,.mp4,.mov,.mkv,.webm" required>
<button type="submit">Generate summary</button>
</form>
{% if summary %}
<hr><h2>Final video summary</h2><div class="result">{{ summary }}</div>
<details><summary>Timestamped transcript</summary><div class="result">{{ transcript }}</div></details>
<details><summary>Visual observations</summary><div class="result">{{ visual }}</div></details>
{% endif %}
</main></body></html>
"""

def get_client():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it in Render Dashboard → Environment.")
    return Groq(api_key=key)

def fmt(seconds):
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}" if h else f"{m:02}:{s:02}"

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Unable to open video. Try an MP4 file encoded with H.264.")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError("Could not read video duration or frame rate.")
    return {"duration": count / fps, "fps": fps, "frame_count": count, "width": width, "height": height}

def extract_audio(path, work):
    pattern = str(work / "audio_%03d.mp3")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path),
           "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
           "-f", "segment", "-segment_time", str(AUDIO_CHUNK_SECONDS),
           "-reset_timestamps", "1", pattern]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return sorted(work.glob("audio_*.mp3"))

def transcribe(client, audio_files, duration):
    parts = []
    for i, audio in enumerate(audio_files):
        start = i * AUDIO_CHUNK_SECONDS
        end = min(start + AUDIO_CHUNK_SECONDS, duration)
        with audio.open("rb") as f:
            result = client.audio.transcriptions.create(model=TRANSCRIPTION_MODEL, file=f)
        parts.append(f"[{fmt(start)} - {fmt(end)}]\n{result.text.strip()}")
    return "\n\n".join(parts) or "[No audio track detected.]"

def extract_frames(path, duration, work):
    cap = cv2.VideoCapture(str(path))
    interval = max(FRAME_INTERVAL_SECONDS, duration / MAX_FRAMES)
    times, t = [], 0.0
    while t < duration and len(times) < MAX_FRAMES:
        times.append(t)
        t += interval
    frames = []
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        if w > 768:
            scale = 768 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        p = work / f"frame_{i:03d}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        frames.append({"timestamp": t, "path": p})
    cap.release()
    return frames

def analyse_frames(client, frames):
    if not frames:
        return "No video frames could be extracted."
    observations = []
    # Groq's currently documented Qwen vision model accepts up to 3 images per request.
    for start in range(0, len(frames), 3):
        batch = frames[start:start + 3]
        content = [{"type": "text", "text":
            "Analyse these video frames in chronological order. Describe visible scene/activity, objects, slides or screens, readable text, demonstrations, and changes. Do not invent facts."}]
        for frame in batch:
            content.append({"type": "text", "text": f"FRAME TIMESTAMP: {fmt(frame['timestamp'])}"})
            with frame["path"].open("rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
        result = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_completion_tokens=1800
        )
        observations.append(result.choices[0].message.content or "")
    return "\n\n".join(observations)

def make_summary(client, transcript, visual):
    prompt = f"""Create an accurate multimodal video summary by combining speech and visual observations. Do not invent facts. Preserve useful timestamps.

TIMESTAMPED TRANSCRIPT:
{transcript[:40000]}

VISUAL OBSERVATIONS:
{visual[:18000]}

Include: VIDEO TITLE; OVERALL SUMMARY (150-250 words if supported); KEY POINTS; TIMESTAMPED CHAPTERS; IMPORTANT VISUAL MOMENTS; KEY TAKEAWAYS; ACTION ITEMS only if present; ONE-SENTENCE SUMMARY."""
    result = client.chat.completions.create(
        model=SUMMARY_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_completion_tokens=3500
    )
    return result.choices[0].message.content or "No summary was returned."

@app.get("/")
def home():
    return render_template_string(PAGE, error=None, summary=None, transcript=None, visual=None)

@app.post("/")
def process_video():
    upload = request.files.get("video")
    if not upload or not upload.filename:
        return render_template_string(PAGE, error="Choose a video file first.", summary=None, transcript=None, visual=None), 400
    name = secure_filename(upload.filename)
    if not name:
        return render_template_string(PAGE, error="Invalid filename.", summary=None, transcript=None, visual=None), 400
    work = WORK_ROOT / uuid.uuid4().hex
    work.mkdir(parents=True, exist_ok=True)
    path = work / name
    try:
        upload.save(path)
        client = get_client()
        info = video_info(path)
        audio_files = extract_audio(path, work)
        transcript = transcribe(client, audio_files, info["duration"])
        frames = extract_frames(path, info["duration"], work)
        visual = analyse_frames(client, frames)
        summary = make_summary(client, transcript, visual)
        return render_template_string(PAGE, error=None, summary=summary, transcript=transcript, visual=visual)
    except subprocess.CalledProcessError:
        app.logger.exception("FFmpeg failed")
        msg = "FFmpeg could not process this video. Try a standard MP4 with H.264 video and AAC audio."
        return render_template_string(PAGE, error=msg, summary=None, transcript=None, visual=None), 400
    except Exception as exc:
        app.logger.exception("Video summarisation failed")
        return render_template_string(PAGE, error=f"Processing failed ({type(exc).__name__}): {exc}", summary=None, transcript=None, visual=None), 500
    finally:
        shutil.rmtree(work, ignore_errors=True)

@app.errorhandler(413)
def too_large(_):
    return "The upload is too large. Reduce the video size or increase MAX_UPLOAD_MB in Render environment settings.", 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
