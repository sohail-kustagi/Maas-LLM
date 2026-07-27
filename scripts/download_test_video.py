import subprocess
import sys
import os

def download_sample_video():
    print("[Downloader] Installing yt-dlp if not present...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yt-dlp", "-q"])
    
    output_path = "tests/disaster_sample.mp4"
    
    if os.path.exists(output_path):
        print(f"[Downloader] Sample video already exists at {output_path}")
        return

    print(f"[Downloader] Searching YouTube for true AERIAL drone flood footage...")
    
    # Use ytsearch to specifically pull a video titled with drone aerial view
    cmd = [
        "yt-dlp", 
        "-f", "best[height<=720]", 
        "--download-sections", "*00:00-00:30", # just grab 30 seconds
        "--match-filter", "duration < 600", # keep it short
        "-o", output_path,
        "ytsearch1:drone aerial flood disaster footage 4k"
    ]
    
    try:
        subprocess.check_call(cmd)
        print(f"[Downloader] ✅ Success! Saved to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[Downloader] ERROR: Failed to download video. {e}")

if __name__ == "__main__":
    os.makedirs("tests", exist_ok=True)
    download_sample_video()
