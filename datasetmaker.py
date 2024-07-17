from minedojo.data import YouTubeDataset
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import random
from pytube import YouTube
from moviepy.editor import VideoFileClip
import os
import time

youtube_dataset = YouTubeDataset(
        full=True,     # full=False for tutorial videos or 
                       # full=True for general gameplay videos
        download=True, # download=True to automatically download data or 
                       # download=False to load data from download_dir
        download_dir="dataset"
                       # default: "~/.minedojo". You can also manually download data from
                       # https://doi.org/10.5281/zenodo.6641142 and put it in download_dir.           
    ) 

index = list(range(len(youtube_dataset)))
random.shuffle(index)




def download_and_process_video(url, output_path='dataset/screenshots', num_screenshots=50, video_num=0):
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    temp_video_path = stream.download()  # Download video temporarily

    clip = VideoFileClip(temp_video_path)
    duration = clip.duration
    intervals = duration / num_screenshots

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(num_screenshots):
        time = intervals * (i + 0.5)  # Capture screenshot at the middle of each interval
        screenshot_path = os.path.join(output_path, f'video_{video_num}_{i+1}.png')
        clip.save_frame(screenshot_path, t=time)

    clip.close()
    os.remove(temp_video_path)  # Delete the temporary video file

for a in index:
    video = youtube_dataset[a]
    url = video['link']
    try:
        download_and_process_video(url, video_num=a)
    except Exception as e:
        print(f"Error processing video {a}: {e}")
        time.sleep(1)
        continue
