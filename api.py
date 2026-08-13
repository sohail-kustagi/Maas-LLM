import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from web_ui import process_video

app = FastAPI(title="MAAS-LLM VOD API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the output videos statically so the frontend can play them
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    # Save uploaded file to temp dir
    temp_dir = "/tmp/maas_vod"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_id = str(uuid.uuid4())
    input_ext = os.path.splitext(video.filename)[1]
    temp_input_path = os.path.join(temp_dir, f"{file_id}{input_ext}")
    
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    try:
        # Run the pipeline synchronously since it binds to the main thread
        # In a production app, we'd use a background task queue (like Celery),
        # but for this VOD use case we'll run it directly.
        gen = process_video(
            video_path=temp_input_path,
            mission_profile_name="search_and_rescue",
            conf_threshold=0.4,
            use_sahi=False,
            for_api=True
        )
        
        out_path = None
        structured_logs = []
        while True:
            try:
                next(gen)
            except StopIteration as e:
                if e.value:
                    out_path, structured_logs = e.value
                break
        
        # Move and convert the output video to the static folder for serving
        # OpenCV uses 'mp4v' which is often unsupported in HTML5 <video>. We must convert to h264.
        static_out_path = os.path.join("static", f"{file_id}_out.mp4")
        if os.path.exists(out_path):
            import subprocess
            subprocess.run(["ffmpeg", "-y", "-i", out_path, "-vcodec", "libx264", static_out_path], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # If ffmpeg failed or wasn't found, fallback to just moving the file
            if not os.path.exists(static_out_path):
                shutil.move(out_path, static_out_path)
            else:
                os.remove(out_path)
            
        return JSONResponse(content={
            "status": "success",
            "video_url": f"http://127.0.0.1:8000/static/{file_id}_out.mp4",
            "logs": structured_logs
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e)
        })
    finally:
        # Clean up temp input file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
