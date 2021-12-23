import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from pathlib import Path
import pandas as pd
from torchvision import datasets
import imageio
import numpy as np
from PIL import Image
import glob
from i3d.pytorch_i3d import InceptionI3d
import os
from torch.autograd import Variable
import i3d.videotransforms
import GPUtil
import time
import math
import pickle


def im2tensor(im, transform=None):
	im = Image.fromarray(im) # convert numpy array to PIL image
	if transform is not None :
		im = transform(im)# Create a PyTorch Variable with the transformed image
	if not isinstance(im, torch.Tensor) :
		tran = transforms.ToTensor()
		im = tran(im)
	return im

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))

def load_frame(frame_file, resize=False):

    data = Image.open(frame_file)

    if resize:
        data = data.resize((224, 224), Image.ANTIALIAS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data

def load_rgb_batch(frames_dir, rgb_files, frame_indices, resize=False):
    if resize:
        batch_data = np.zeros(frame_indices.shape + (224,224,3))
    else:
        batch_data = np.zeros(frame_indices.shape + (256,340,3))

    for i in range(frame_indices.shape[0]):
            #print("Loading frame : ", os.path.join(frames_dir,rgb_files[frame_indices[i]]))
            batch_data[i,:,:,:] = load_frame(os.path.join(frames_dir, 
                rgb_files[frame_indices[i]]), resize)
    return batch_data

class VideoDataset_EpicKitchens(Dataset):
	'''
    Input : 
	    csv_file     : Path to file where path to videos is stored - <path>, label
        frequency    : Sampling frequency for the i3d.
        num_nodes    : Number of graph nodes. Set to 16
        is_test      : Test/Train
	    transform    : if specified applies the transform to the frames.
        base_dir     : Base directory to the dataset. Please follow the instructions mentioned in the ReadMe
    
    Returns : 
        Tensor : [num_nodes, C, 8, H, W] ### num_nodes clips each consisting of 8 consecutive frames: 
        label  : Label of the video. (Not used for target dataset during training)    

	'''
	def __init__(self, csv_file, frequency = 4, num_nodes = 16, is_test = False, transform = None, base_dir='./data/epic_kitchens'):
		self.transform = transform
		with open(csv_file, 'rb') as f:
			dataset_pd = pickle.load(f)
		self.csv_file = csv_file
		self.uid = dataset_pd["uid"].to_numpy()
		self.start_frame = dataset_pd["start_frame"].to_numpy()
		self.stop_frame = dataset_pd["stop_frame"].to_numpy()
		self.video_id = dataset_pd["video_id"].to_numpy()
		self.verb_class = dataset_pd["verb_class"].to_numpy()

		self.min_frames = 72
		self.frequency = frequency
		self.chunk_size = 8
		self.num_nodes = num_nodes
		self.is_test = is_test
		if not base_dir.endswith("/"):
			base_dir += "/"
		self.video_dir = base_dir + "epic_kitchens_videos/"
		if self.csv_file[:-4].endswith("train"):
			self.video_dir += "train/"
		else : 
			assert(self.is_test == True)
			self.video_dir += "test/"
		
		stripped_csv_file = self.csv_file.split("/")[-1]
		if stripped_csv_file.startswith("D1"):
			self.video_dir += "D1/"
		if stripped_csv_file.startswith("D2"):
			self.video_dir += "D2/"
		if stripped_csv_file.startswith("D3"):
			self.video_dir += "D3/"

	def __len__(self):
		return len(self.uid)

	def __getitem__(self, idx) :
		path = self.video_dir + self.video_id[idx]
		label = self.verb_class[idx] 
		if not self.is_test:
			bg_path = path.replace("epic_kitchens_videos", "epic_kitchens_BG") + "_" + str(self.uid[idx])
			bg_rgb_files = [i for i in os.listdir(bg_path)]
			bg_rgb_files.sort()
			bg_frame_indices = np.arange(len(bg_rgb_files))

		rgb_files = [i for i in os.listdir(path)]
		rgb_files.sort()
		rgb_files = rgb_files[self.start_frame[idx]:self.stop_frame[idx]]

		frame_indices = np.arange(len(rgb_files))
		num_frames = len(rgb_files)
		if num_frames == 0:
			print("No images found inside the directory : ", path)
			raise Exception
		frames_tensor = load_rgb_batch(path, rgb_files, frame_indices, resize=True)
		if not self.is_test:
			bg_frames_tensor = load_rgb_batch(bg_path, bg_rgb_files, bg_frame_indices, resize=True)

		if self.transform and not self.is_test : 
			frames_tensor = self.transform(frames_tensor)

		frames_tensor = video_to_tensor(frames_tensor) # [C,T,H,W] pytorch tensor
		if not self.is_test:
			bg_frames_tensor = video_to_tensor(bg_frames_tensor) # [C,T,H,W] pytorch tensor

		if num_frames < self.min_frames :
			frames_tensor = torch.repeat_interleave(frames_tensor, math.ceil(self.min_frames/frames_tensor.shape[1]), dim=1)

		max_num_feats = frames_tensor.shape[1] // self.frequency - math.ceil(self.chunk_size/self.frequency) # ith feature is [i*frequency, i*frequency + chunk_size]	
		allRange = np.arange(max_num_feats)
		splitRange = np.array_split(allRange, self.num_nodes)
		try: 
			if not self.is_test : 
				fidx = [np.random.choice(a) for a in splitRange]
			else : 
				fidx = [a[0] for a in splitRange]
		except:
			print("Path : ", path)
			print("Split range : ", splitRange)
			print("All range : ", allRange)
			raise Exception
			
		ind = [np.arange(start=i*self.frequency, stop=i*self.frequency + self.chunk_size, step=1) for i in fidx]	
		frames_tensor_chunks = torch.empty(self.num_nodes, frames_tensor.shape[0], self.chunk_size, frames_tensor.shape[2], frames_tensor.shape[3]) # [16, C, chunk_size, H, W]	
		for chunk_ind, i in zip(ind, range(self.num_nodes)) : 
			frames_tensor_chunks[i, :, :, :, :] = frames_tensor[:, chunk_ind, :, :]

		if self.is_test:
			bg_frames_tensor = 'None'

		return [frames_tensor_chunks, bg_frames_tensor], label # List of tensors, label

class VideoDataset_Jester(Dataset):
	'''
    Input : 
	    csv_file     : Path to file where path to videos is stored - <path>, label
        frequency    : Sampling frequency for the i3d.
        num_nodes    : Number of graph nodes. Set to 16
        is_test      : Test/Train
	    transform    : if specified applies the transform to the frames.
        base_dir     : Base directory to the dataset. Please follow the instructions mentioned in the ReadMe
    
    Returns : 
        Tensor : [num_nodes, C, 8, H, W] ### num_nodes clips each consisting of 8 consecutive frames: 
        label  : Label of the video. (Not used for target dataset during training)    

	'''
	def __init__(self, csv_file, frequency = 4, num_nodes = 16, is_test = False, transform = None, base_dir='./data/jester'):
		self.transform = transform
		self.dataset = pd.read_csv(csv_file, header=None)
		self.min_frames = 72
		self.frequency = frequency
		self.chunk_size = 8
		self.num_nodes = num_nodes
		self.is_test = is_test
		if not base_dir.endswith("/"):
			base_dir += "/"
		self.bg_dir = base_dir + "jester_BG/"
		self.video_dir = base_dir + "jester_videos/"
	def __len__(self):
		return len(self.dataset)
	def __getitem__(self, idx) :
		path = self.video_dir + str(self.dataset.iloc[idx, 0])
		label = self.dataset.iloc[idx, 1]
		bg_path = self.bg_dir + '/' + path.split('/')[-1]
		rgb_files = [i for i in os.listdir(path)]
		bg_rgb_files = [i for i in os.listdir(bg_path)]
		rgb_files.sort()
		bg_rgb_files.sort()

		frame_indices = np.arange(len(rgb_files))
		bg_frame_indices = np.arange(len(bg_rgb_files))
		num_frames = len(rgb_files)
		if num_frames == 0:
			print("No images found inside the directory : ", path)
			raise Exception
		frames_tensor = load_rgb_batch(path, rgb_files, frame_indices, resize=True)
		bg_frames_tensor = load_rgb_batch(bg_path, bg_rgb_files, bg_frame_indices, resize=True)

		if self.transform and not self.is_test : 
			frames_tensor = self.transform(frames_tensor)

		frames_tensor = video_to_tensor(frames_tensor) # [C,T,H,W] pytorch tensor
		bg_frames_tensor = video_to_tensor(bg_frames_tensor) # [C,T,H,W] pytorch tensor

		if num_frames < self.min_frames :
			frames_tensor = torch.repeat_interleave(frames_tensor, math.ceil(self.min_frames/frames_tensor.shape[1]), dim=1)

		max_num_feats = frames_tensor.shape[1] // self.frequency - math.ceil(self.chunk_size/self.frequency) # ith feature is [i*frequency, i*frequency + chunk_size]	
		allRange = np.arange(max_num_feats)
		splitRange = np.array_split(allRange, self.num_nodes)
		try: 
			if not self.is_test : 
				fidx = [np.random.choice(a) for a in splitRange]
			else : 
				fidx = [a[0] for a in splitRange]
		except:
			print("Path : ", path)
			print("Split range : ", splitRange)
			print("All range : ", allRange)
			raise Exception
			
		ind = [np.arange(start=i*self.frequency, stop=i*self.frequency + self.chunk_size, step=1) for i in fidx]	
		frames_tensor_chunks = torch.empty(self.num_nodes, frames_tensor.shape[0], self.chunk_size, frames_tensor.shape[2], frames_tensor.shape[3]) # [16, C, chunk_size, H, W]	
		#print("Final size : ", frames_tensor_chunks.shape)
		for chunk_ind, i in zip(ind, range(self.num_nodes)) : 
			#print("Iteration : ", i, " Chunk indices : ", chunk_ind, frames_tensor[:,chunk_ind,:,:].shape)
			frames_tensor_chunks[i, :, :, :, :] = frames_tensor[:, chunk_ind, :, :]

		return [frames_tensor_chunks, bg_frames_tensor], label # List of tensors, label

class VideoDataset_UCFHMDB(Dataset):
	'''
    Input : 
	    csv_file     : Path to file where path to videos is stored - <path>, label
        frequency    : Sampling frequency for the i3d.
        num_nodes    : Number of graph nodes. Set to 16
        is_test      : Test/Train
	    transform    : if specified applies the transform to the frames.
        dataset_name : Name of the dataset
        base_dir     : Base directory to the dataset. Please follow the instructions mentioned in the ReadMe
    
    Returns : 
        Tensor : [num_nodes, C, 8, H, W] ### num_nodes clips each consisting of 8 consecutive frames: 
        label  : Label of the video. (Not used for target dataset during training)    

	'''
	def __init__(self, csv_file, frequency = 4, num_nodes = 16, is_test = False, transform = None, dataset_name='ucf', base_dir='./data/ucf_hmdb'):
		self.transform = transform
		self.dataset = pd.read_csv(csv_file, header=None)
		self.min_frames = 72
		self.frequency = frequency
		self.chunk_size = 8
		self.num_nodes = num_nodes
		self.is_test = is_test
		if not base_dir.endswith("/"):
			base_dir += "/"
		self.bg_dir = base_dir + str(dataset_name) + "_BG/"
		self.video_dir = base_dir + str(dataset_name) + "_videos/" 
	def __len__(self):
		return len(self.dataset)
	def __getitem__(self, idx) :
		path = self.video_dir +  self.dataset.iloc[idx, 0]
		label = self.dataset.iloc[idx, 1]
		bg_path = self.bg_dir + path.split('/')[-1]
		rgb_files = [i for i in os.listdir(path)]
		bg_rgb_files = [i for i in os.listdir(bg_path)]

		rgb_files.sort()
		bg_rgb_files.sort()

		frame_indices = np.arange(len(rgb_files))
		bg_frame_indices = np.arange(len(bg_rgb_files))
		num_frames = len(rgb_files)
		if num_frames == 0:
			print("No images found inside the directory : ", path)
			raise Exception
		frames_tensor = load_rgb_batch(path, rgb_files, frame_indices, resize=True)
		bg_frames_tensor = load_rgb_batch(bg_path, bg_rgb_files, bg_frame_indices, resize=True)

		if self.transform and not self.is_test : 
			frames_tensor = self.transform(frames_tensor)

		frames_tensor = video_to_tensor(frames_tensor) # [C,T,H,W] pytorch tensor
		bg_frames_tensor = video_to_tensor(bg_frames_tensor) # [C,T,H,W] pytorch tensor

		if num_frames < self.min_frames :
			frames_tensor = torch.repeat_interleave(frames_tensor, math.ceil(self.min_frames/frames_tensor.shape[1]), dim=1)

		max_num_feats = frames_tensor.shape[1] // self.frequency - math.ceil(self.chunk_size/self.frequency) # ith feature is [i*frequency, i*frequency + chunk_size]	
		allRange = np.arange(max_num_feats)
		splitRange = np.array_split(allRange, self.num_nodes)
		try: 
			if not self.is_test : 
				fidx = [np.random.choice(a) for a in splitRange]
			else : 
				fidx = [a[0] for a in splitRange]
		except:
			print("Path : ", path)
			print("Split range : ", splitRange)
			print("All range : ", allRange)
			raise Exception
			
		ind = [np.arange(start=i*self.frequency, stop=i*self.frequency + self.chunk_size, step=1) for i in fidx]	
		frames_tensor_chunks = torch.empty(self.num_nodes, frames_tensor.shape[0], self.chunk_size, frames_tensor.shape[2], frames_tensor.shape[3]) # [16, C, chunk_size, H, W]	
		for chunk_ind, i in zip(ind, range(self.num_nodes)) : 
			frames_tensor_chunks[i, :, :, :, :] = frames_tensor[:, chunk_ind, :, :]

		return [frames_tensor_chunks, bg_frames_tensor], label # List of tensors, label
		
