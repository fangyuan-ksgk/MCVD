import pytube
from pytube import YouTube
import os
import cv2
import numpy as np
import tqdm
from tqdm import tqdm as tqdm
# Minimal Implementation of the Citiscape dataset object -- basically just getting the consecutive frames from the images
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob

def extract_frames_from_video(video_path, output_dir, frame_interval=1):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_capture = cv2.VideoCapture(video_path)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    for frame_number in tqdm(range(0, frame_count, frame_interval)):
        success, frame = video_capture.read()
        if not success:
            break

        frame_file_name = f"{video_name}_frame_{frame_number:04d}.jpg"
        frame_output_path = os.path.join(output_dir, frame_file_name)
        cv2.imwrite(frame_output_path, frame)

    video_capture.release()

    print(f"Frames extracted from {video_name} successfully.")


def download_n_preprocess_youtube_video(url, vid_name, save_dir, resolution='360p', frame_interval=1):
    # Automatic Section of the data preparation pipeline
    y_vid = YouTube(url)
    print('----- Title of the Video from Youtube -----')
    print(y_vid.title)
    print('----- Tumbnail Image ------')
    print(y_vid.thumbnail_url)
    res_ok = False
    for stream in y_vid.streams:
        if stream.resolution==resolution:
            res_ok = True
            break
    assert res_ok, 'We prefer 360p video but current stream is not -- check manually on various video config !'
    
    os.makedirs(save_dir, exist_ok=True)
    stream.download(output_path=save_dir, filename=f'{vid_name}.mp4')
    print('Video Download Complete! : ', vid_name)

    # Clean up and prepare images data frame output
    video_path = save_dir + '/' + vid_name + '.mp4'
    frame_dir = save_dir + '/' + vid_name
    os.makedirs(frame_dir, exist_ok=True)
    extract_frames_from_video(video_path, frame_dir, frame_interval=frame_interval)
    return

# Collect Video Information (downloaded video from pytube)
def update_vid_info(vid_info, vid_name, save_dir):
    frames = glob.glob(save_dir + f'/{vid_name}/*.jpg')
    video_len = len(frames)
    info = {'name':vid_name, 'length': video_len, 'frames':frames}
    vid_info.append(info)

def get_vid_infos(save_dir):
    # get vid names
    vid_names = [p[:-4] for p in os.listdir(save_dir) if p.endswith('mp4')]
    vid_info = []
    for vid_name in vid_names:
        update_vid_info(vid_info, vid_name, save_dir)
    return vid_info

# separate index to point to video & time index
def separate_index(index, len_accessible):
    assert index<np.sum(len_accessible), f"index {index} too big for accesbile frame number {np.sum(len_accessible)}"
    i = 0
    prev_sum = 0
    curr_sum = 0
    while i<len(len_accessible):
        curr_sum = prev_sum + len_accessible[i]
        if index < curr_sum: # index belongs to the current video frames
            video_index = i
            time_index = index - prev_sum
            break
        else:
            prev_sum = curr_sum
            i += 1
    return video_index, time_index


# There are some redundant frames with a giant title in the middle of the video 
class YoutubeVideoDataset(Dataset):
    def __init__(self, save_dir, frames_per_sample=5, random_horizontal_flip=True, color_jitter=0):
        self.frames_per_sample = frames_per_sample
        self.random_horizontal_flip = random_horizontal_flip
        self.color_jitter = color_jitter
        self.jitter = transforms.ColorJitter(hue=color_jitter)
        # TBD: compatible with multiple video input
        self.save_dir = save_dir
        self.vid_info = get_vid_infos(self.save_dir)
        self.len_accessible = [info['length']-self.frames_per_sample for info in self.vid_info]
        self.total_len_accessible = np.sum(self.len_accessible)
        
    def __len__(self):
        return self.total_len_accessible        

    def window_stack(self, a, width=3, step=1):
        return torch.stack([a[i:1+i-width or None:step] for i in range(width)]).transpose(0, 1)

    def __getitem__(self, index):
        # separate index into vid_index and time_index here
        video_idx, time_idx = separate_index(index, self.len_accessible)
        frames = self.vid_info[video_idx]['frames']
        prefinals = []
        flip_p = np.random.randint(2) == 0 if self.random_horizontal_flip else 0
        for i in range(time_idx, time_idx+self.frames_per_sample):
            img = Image.open(frames[i])
            arr = transforms.RandomHorizontalFlip(flip_p)(transforms.ToTensor()(img))
            prefinals.append(arr)
        data = torch.stack(prefinals)
        data = self.jitter(data)
        return data

