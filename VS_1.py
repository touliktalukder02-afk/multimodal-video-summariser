# ============================================================
# MULTIMODAL VIDEO SUMMARISATION USING OPENAI
#
# Pipeline:
#
# VIDEO
#   |
#   +----> AUDIO ----> Speech-to-Text
#   |
#   +----> FRAMES ---> Vision Analysis (OCR)
#                         |
#                         v
#              MULTIMODAL FUSION
#                         |
#                         v
#                VIDEO SUMMARY
#
# ============================================================

import os
import cv2
import math
import base64
import subprocess
from pathlib import Path
from openai import OpenAI


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Paste your OpenAI API key here
#GROQ_API_KEY

#OPENAI_API_KEY = ""
#client = OpenAI(api_key=OPENAI_API_KEY)

# Input video
VIDEO_PATH = "AI10.mp4"

# Models
TRANSCRIPTION_MODEL = "whisper-large-v3"

VISION_MODEL = "openai/gpt-oss-120b"

SUMMARY_MODEL = "openai/gpt-oss-120b"


# Extract at most this many frames
MAX_FRAMES = 15

# Try to sample approximately every 30 seconds
FRAME_INTERVAL_SECONDS = 30

# Split speech into 60-second chunks
AUDIO_CHUNK_SECONDS = 60


# Output directories
FRAME_DIR = Path("video_frames")
AUDIO_DIR = Path("audio_chunks")

FRAME_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. HELPER: CONVERT SECONDS TO HH:MM:SS
# ============================================================

def format_time(seconds):

    seconds = int(seconds)

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{hours:02}:{minutes:02}:{secs:02}"

    return f"{minutes:02}:{secs:02}"


# ============================================================
# 3. GET VIDEO INFORMATION
# ============================================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Unable to open video.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height
    }


print("\n================ VIDEO INFORMATION ================\n")

video_info = get_video_info(VIDEO_PATH)

for key, value in video_info.items():
    print(f"{key}: {value}")

duration = video_info["duration"]

print("\nVideo Duration:", format_time(duration))


# ============================================================
# 4. AUDIO MODALITY
# ============================================================
#
# Video
#   |
# Extract Audio
#   |
# Split audio into chunks
#   |
# OpenAI Speech-to-Text
#   |
# Timestamped transcript
#
# ============================================================


def extract_audio_chunks(video_path):

    print("\n================ AUDIO EXTRACTION ================\n")

    output_pattern = str(AUDIO_DIR / "chunk_%03d.mp3")

    command = [
        "ffmpeg",
        "-y",
        "-i", video_path,

        # Remove video
        "-vn",

        # Mono audio
        "-ac", "1",

        # 16 kHz
        "-ar", "16000",

        # Compress audio
        "-b:a", "64k",

        # Split audio
        "-f", "segment",

        "-segment_time", str(AUDIO_CHUNK_SECONDS),

        "-reset_timestamps", "1",

        output_pattern
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    audio_files = sorted(AUDIO_DIR.glob("chunk_*.mp3"))

    print("Audio chunks created:", len(audio_files))

    return audio_files


audio_files = extract_audio_chunks(VIDEO_PATH)


# ============================================================
# 5. TRANSCRIBE AUDIO
# ============================================================


def transcribe_audio(audio_files, video_duration):

    print("\n================ SPEECH TRANSCRIPTION ================\n")

    transcript_segments = []

    for index, audio_file in enumerate(audio_files):

        start_time = index * AUDIO_CHUNK_SECONDS

        end_time = min(
            start_time + AUDIO_CHUNK_SECONDS,
            video_duration
        )

        print(
            f"Transcribing: "
            f"{format_time(start_time)} - "
            f"{format_time(end_time)}"
        )

        with open(audio_file, "rb") as file:

            transcription = client.audio.transcriptions.create(
                model=TRANSCRIPTION_MODEL,
                file=file
            )

        text = transcription.text.strip()

        transcript_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text
        })

    return transcript_segments


transcript_segments = transcribe_audio(
    audio_files,
    duration
)


# ============================================================
# 6. CREATE TIMESTAMPED TRANSCRIPT
# ============================================================


def create_timestamped_transcript(segments):

    transcript = ""

    for segment in segments:

        transcript += (
            f"[{format_time(segment['start'])}"
            f" - "
            f"{format_time(segment['end'])}]\n"
            f"{segment['text']}\n\n"
        )

    return transcript


timestamped_transcript = create_timestamped_transcript(
    transcript_segments
)


print("\n================ TRANSCRIPT ================\n")

print(timestamped_transcript)


# Save transcript
with open(
    "video_transcript.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(timestamped_transcript)


# ============================================================
# 7. VISUAL MODALITY
# ============================================================
#
# Video
#   |
# Sample representative frames
#   |
# Timestamp each frame
#   |
# Send frames to GPT vision model
#
# ============================================================


def extract_frames(video_path, duration):

    print("\n================ FRAME EXTRACTION ================\n")

    cap = cv2.VideoCapture(video_path)

    # Dynamic interval
    #
    # Example:
    # 5 minute video -> approx every 30 sec
    #
    # 2 hour video -> increase interval so we don't
    # send hundreds of images.

    interval = max(
        FRAME_INTERVAL_SECONDS,
        duration / MAX_FRAMES
    )

    timestamps = []

    current_time = 0

    while current_time < duration:

        timestamps.append(current_time)

        current_time += interval

    timestamps = timestamps[:MAX_FRAMES]

    extracted_frames = []

    for index, timestamp in enumerate(timestamps):

        # Move to timestamp
        cap.set(
            cv2.CAP_PROP_POS_MSEC,
            timestamp * 1000
        )

        success, frame = cap.read()

        if not success:
            continue

        # Resize large video frames to reduce API cost
        height, width = frame.shape[:2]

        max_width = 768

        if width > max_width:

            scale = max_width / width

            frame = cv2.resize(
                frame,
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        filename = (
            FRAME_DIR /
            f"frame_{index:03d}_{int(timestamp)}sec.jpg"
        )

        cv2.imwrite(
            str(filename),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 75]
        )

        extracted_frames.append({
            "timestamp": timestamp,
            "path": str(filename)
        })

        print(
            f"Frame {index + 1}: "
            f"{format_time(timestamp)}"
        )

    cap.release()

    return extracted_frames


frames = extract_frames(
    VIDEO_PATH,
    duration
)

print("\nTotal frames extracted:", len(frames))


# ============================================================
# 8. CONVERT IMAGE TO BASE64
# ============================================================


def image_to_base64(image_path):

    with open(image_path, "rb") as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


# ============================================================
# 9. GPT VISUAL UNDERSTANDING
# ============================================================
#
# Multiple frames are sent to GPT.
#
# GPT understands:
#
# - scenes
# - objects
# - slides
# - diagrams
# - activities
# - people
# - screen content
# - visual transitions
#
# ============================================================


def analyse_video_frames(frames):

    print("\n================ VISUAL ANALYSIS ================\n")

    content = []

    content.append({
        "type": "input_text",
        "text": """
You are analysing representative frames extracted
from a video.

Study the frames in chronological order.

For each timestamp identify:

1. Main scene or activity
2. Important visible objects
3. Slide/screen/topic being shown
4. Important visible text if readable
5. Demonstrations or actions
6. Scene changes
7. Information useful for summarising the video

Do NOT invent information that is not visible.

Return observations organised by timestamp.
"""
    })


    for frame in frames:

        timestamp = format_time(
            frame["timestamp"]
        )

        # Add timestamp
        content.append({
            "type": "input_text",
            "text": f"\nFRAME TIMESTAMP: {timestamp}"
        })

        # Encode image
        base64_image = image_to_base64(
            frame["path"]
        )

        content.append({
            "type": "input_image",
            "image_url":
            f"data:image/jpeg;base64,{base64_image}"
        })


    response = client.responses.create(

        model=VISION_MODEL,

        input=[
            {
                "role": "user",
                "content": content
            }
        ]
    )

    return response.output_text


visual_analysis = analyse_video_frames(
    frames
)


print("\n================ VISUAL OBSERVATIONS ================\n")

print(visual_analysis)


with open(
    "visual_analysis.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(visual_analysis)


# ============================================================
# 10. MULTIMODAL FUSION
# ============================================================
#
#               SPEECH
#                  |
#                  v
#             Transcript
#                  |
#                  |
# VIDEO ---> Visual Analysis
#                  |
#                  v
#          Multimodal Fusion
#                  |
#                  v
#            Final Summary
#
# ============================================================


def create_multimodal_summary(
    transcript,
    visual_analysis
):

    print("\n================ MULTIMODAL FUSION ================\n")

    prompt = f"""
You are an expert multimodal video summarisation system.

You have information from TWO MODALITIES.

====================================================
MODALITY 1: AUDIO / SPEECH
====================================================

The following is the timestamped speech transcript:

{transcript}


====================================================
MODALITY 2: VISUAL
====================================================

The following observations were produced by analysing
representative video frames:

{visual_analysis}


====================================================
TASK
====================================================

Fuse BOTH modalities to generate an accurate video summary.

Important:

- Speech tells you what is being discussed.
- Visual information tells you what is being shown.
- Combine information only when supported by evidence.
- Do not invent facts.
- Preserve useful timestamps.

Generate the following sections:

# VIDEO TITLE

Suggest a suitable title.

# OVERALL SUMMARY

Write a concise 150-250 word summary.

# KEY POINTS

Give the most important points discussed.

# TIMESTAMPED CHAPTERS

Create chapters in this format:

00:00 - 02:30
Chapter title
2-3 sentence explanation

# IMPORTANT VISUAL MOMENTS

Mention important slides, diagrams, demonstrations,
screens, objects, activities, or scene changes.

# KEY TAKEAWAYS

Give 5-10 major takeaways.

# ACTION ITEMS

Mention action items only if they genuinely appear
in the video.

# ONE-SENTENCE SUMMARY

Summarise the entire video in one sentence.
"""


    response = client.responses.create(

        model=SUMMARY_MODEL,

        input=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.output_text


final_summary = create_multimodal_summary(
    timestamped_transcript,
    visual_analysis
)


# ============================================================
# 11. DISPLAY FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("FINAL MULTIMODAL VIDEO SUMMARY")
print("=" * 70)
print("\n")

print(final_summary)


# ============================================================
# 12. SAVE FINAL RESULT
# ============================================================

with open(
    "multimodal_video_summary.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(final_summary)


print("\n")
print("=" * 70)
print("PROCESS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("1. video_transcript.txt")
print("2. visual_analysis.txt")
print("3. multimodal_video_summary.txt")
print("4. video_frames/")