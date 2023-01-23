import os
import os.path as P
import ffmpeg
import json
import tqdm
import numpy as np
import threading
import time
import multiprocessing
from multiprocessing import Pool
import subprocess



### change diffenrent datasets
input_path = '../datasets/msrvtt/raw_videos'
output_path = '../datasets/msrvtt/testt'
data_list = os.listdir(input_path)

def execCmd(cmd):
    r = os.popen(cmd)
    text = r.read()
    r.close()
    return text

def pipline(video_path, video_probe, output_dir, fps, sr, duration_target):
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace(".mp4", ".wav")
    video_name = video_name.replace(".mp4", "")


    #extract video frames fps
    fps_frame_dir = P.join(output_dir, f"frames_fps{fps}", video_name)
    os.makedirs(fps_frame_dir, exist_ok=True)
    cmd = "ffmpeg -loglevel error -i {} -vsync 0 -f image2 -vf fps=fps={:.02f} -qscale:v 2 {}/frame_%04d.jpg".format(
              video_path, fps, fps_frame_dir)
    subprocess.call(cmd, shell=True)

    # Extract Audio
    sr_audio_dir = P.join(output_dir,f"audio_{sr}hz")
    os.makedirs(sr_audio_dir, exist_ok=True)
    audio_file_path = P.join(sr_audio_dir, audio_name)
    cmd = "ffmpeg -i {} -loglevel error -f wav -vn -ac 1 -ab 16k -ar {} -y {}".format(
            video_path, sr, audio_file_path)
    subprocess.call(cmd, shell=True)


def extract_thread(video_id):
    
    video_name = os.path.join(input_path, video_id)
    if not os.path.exists(video_name):
        return
    probe = ffmpeg.probe(video_name)
    pipline(video_name, probe, output_path, fps=4, sr=22050, duration_target=10)


def extract_all(video_ids, thread_num, start):
    length = len(video_ids)
    print(length)
    with Pool(thread_num) as p:
        list(tqdm.tqdm(p.imap(extract_thread, video_ids), total=length))

if __name__=='__main__':
    thread_num = 50
    start = 0

    print(len(data_list))
    extract_all(data_list, thread_num, start)

