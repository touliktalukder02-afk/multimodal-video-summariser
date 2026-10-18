MULTIMODAL VIDEO SUMMARISER — RENDER DEPLOYMENT

Files:
- app.py: web app for video upload, audio transcription, visual frame analysis and summary
- requirements.txt: Python packages
- Dockerfile: installs FFmpeg and starts Gunicorn
- render.yaml: Render Blueprint configuration

DEPLOY:
1. Create a GitHub repository and upload all four files.
2. Open Render and sign in.
3. Choose New + -> Blueprint.
4. Connect your GitHub repository and select the repository containing render.yaml.
5. When asked for GROQ_API_KEY, provide your Groq API key as a secret environment variable. Never put it in app.py or commit it to GitHub.
6. Deploy. Wait for build and deploy to finish, then open the Render URL.
7. Upload a short MP4 first to test the app.

MODEL NOTES:
- Default transcription: whisper-large-v3
- Default vision: qwen/qwen3.8-27b (Groq's documented vision-capable model; the code sends at most 3 frames per request)
- Default summary: llama-3.3-70b-versatile
Check Groq's current model catalogue if a model is unavailable to your account; change the Render environment variable accordingly.

LIMITS:
- Processing time and API usage depend on video length and number of frames.
- The web request may exceed some hosting-plan time limits for long videos. Start with a short video.
- Files are held temporarily during processing and removed afterward; this app does not save uploaded videos or results persistently.
- Do not upload private/sensitive video unless you have permission to process it.
