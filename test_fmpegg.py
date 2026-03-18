import subprocess

FFMPEG_PATH = r"C:\Users\Roshini\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe"

subprocess.run([FFMPEG_PATH, "-version"])