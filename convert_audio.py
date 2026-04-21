#!/usr/bin/env python3
"""Convert MP4 audio files to WAV format"""

import os
import subprocess
from pathlib import Path

# Try using ffmpeg via Windows Media Feature Pack or download it
def convert_with_ffmpeg_exe():
    """Try to find ffmpeg.exe in common locations"""
    common_paths = [
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return None

# Try downloading and using ffmpeg via Python
def convert_with_moviepy():
    """Use moviepy to convert MP4 to WAV"""
    try:
        from moviepy.editor import AudioFileClip
        print("Using moviepy to convert audio files...")
        
        music_dir = Path(__file__).parent / 'Music'
        mp4_files = [
            'BACKGROUND MUSIC.mp4',
            'STAGE FIGHT.mp4',
            'WIN EFECT.mp4'
        ]
        
        for mp4_file in mp4_files:
            mp4_path = music_dir / mp4_file
            wav_path = music_dir / mp4_file.replace('.mp4', '.wav')
            
            if mp4_path.exists():
                print(f"Converting {mp4_file}...")
                try:
                    audio = AudioFileClip(str(mp4_path))
                    audio.write_audiofile(str(wav_path), verbose=False, logger=None)
                    print(f"✓ Converted: {mp4_file} → {wav_path.name}")
                except Exception as e:
                    print(f"✗ Failed to convert {mp4_file}: {e}")
            else:
                print(f"File not found: {mp4_path}")
        
        return True
    except ImportError:
        return False

def main():
    # Try ffmpeg first
    ffmpeg_path = convert_with_ffmpeg_exe()
    
    if ffmpeg_path:
        print(f"Found FFmpeg at: {ffmpeg_path}")
        music_dir = Path(__file__).parent / 'Music'
        
        conversions = [
            ('BACKGROUND MUSIC.mp4', 'BACKGROUND MUSIC.wav'),
            ('STAGE FIGHT.mp4', 'STAGE FIGHT.wav'),
            ('WIN EFECT.mp4', 'WIN EFECT.wav'),
        ]
        
        for mp4_file, wav_file in conversions:
            mp4_path = music_dir / mp4_file
            wav_path = music_dir / wav_file
            
            if mp4_path.exists():
                print(f"\nConverting {mp4_file}...")
                cmd = [
                    ffmpeg_path,
                    '-i', str(mp4_path),
                    '-acodec', 'pcm_s16le',
                    '-ar', '22050',
                    str(wav_path)
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True)
                    print(f"✓ Converted: {wav_file}")
                except subprocess.CalledProcessError as e:
                    print(f"✗ FFmpeg error: {e}")
    else:
        print("FFmpeg not found in standard locations.")
        print("Attempting to use moviepy...")
        
        if not convert_with_moviepy():
            print("\n❌ Failed! Install moviepy: pip install moviepy")
            print("Or download FFmpeg from: https://ffmpeg.org/download.html")

if __name__ == '__main__':
    main()
