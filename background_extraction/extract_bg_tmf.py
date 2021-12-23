from __future__ import print_function

import argparse
import numpy
import os
import glob
import sys
import numpy as np
import pandas as pd

from timeit import default_timer as timer
from multiprocessing import Process, Pool


from PIL import Image
import imageio

IMAGE_EXTENSIONS = ["*.tiff", "*.tif", "*.jpeg", "*.png", "*.jpg", "*.dng"]
MOVIE_EXTENSIONS = ["*.mp4", "*.mov"]


def get_frame_data(input_data, frame_number):
    """ Gets the frame with the specified index """
    if isinstance(input_data, list):
        #print(frame_number)
        return Image.open(input_data[frame_number])
    return input_data.get_data(frame_number)


def get_number_of_frames(input_data):
    if isinstance(input_data, list):
        return len(input_data)
    return round(input_data.get_meta_data()['fps'] * input_data.get_meta_data()['duration']) -1


def make_a_glob(root_dir):
    """ Creates a glob of images from specified path. Checks for JPEG, PNG, TIFF, DNG

    Args:
        root_dir (str): path to a video or directory of images

    Returns:
        (list, imageio.core.format.Reader): glob of images from input path
    """
    # video file, return here
    if os.path.isfile(root_dir):
        return imageio.get_reader(root_dir)

    # start hunting for an image sequence
    if not os.path.exists(root_dir):
        raise IOError("No such path: %s" % root_dir)

    if not root_dir.endswith("/"):
        root_dir += "/"

    root_dir = glob.escape(root_dir)
    input_data = glob.glob(root_dir + "*.tif")

    for ext in IMAGE_EXTENSIONS:
        if len(input_data) == 0:
            input_data = glob.glob(root_dir + ext)
            if len(input_data) == 0:
                input_data = glob.glob(root_dir + ext.upper())
            if ext == IMAGE_EXTENSIONS[(len(IMAGE_EXTENSIONS)-1)] and len(input_data) == 0:
                raise IOError("No images found in directory: %s" % root_dir)
        else:
            break
    input_data.sort()
    print("First image is: " + input_data[0])
    print("Number of frames found: ", len(input_data))
    return input_data


def get_frame_limit(limit_frames, globsize):
    """ Determines a limit on the number of frames

    Args:
        limit_frames:
        globsize:

    Returns:
        int: total frames to run TMF on
    """
    if limit_frames != -1:
        if globsize > limit_frames:
            total_frames = limit_frames
            print("Frames limited to ", limit_frames)
        else:
            print("Frame limit of ", limit_frames, "is higher than total # of frames: ", globsize)
            total_frames = globsize
    else:
        total_frames = globsize

    return total_frames


def make_output_dir(output_dir, input_dir):
    """ Creates uniquely-named new folder along specified path

    Args:
        output_dir: output path

    Returns:
        str: path to new folder
    """
    if not output_dir.endswith("/"):
        output_dir += "/"

    output_path = output_dir
    if not(os.path.exists(output_path) and os.path.isdir(output_path)):
        print("Made directory: ", output_path)
        os.makedirs(output_path)
    frame_path = output_path

    return frame_path


def do_sizing(input_data):
    if isinstance(input_data, list):
        first = Image.open(input_data[0])
        width, height = first.size
        print("width is: ", width, " height is: ", height)
    else:
        return input_data.get_meta_data()['size']
    return width, height


def progress(count, total, suffix=''):
    """ Creates and displays a progress bar in console log.

    Args:
        count: parts completed
        total: parts to complete
        suffix: any additional descriptors
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def temporal_median_filter_multi2(input_data, output_dir, limit_frames, output_format, frame_offset=8, resize=None, input_dir=None):
    start2 = timer()
    frame_path = make_output_dir(output_dir, input_dir)
    width, height = do_sizing(input_data)
    total_frames = get_frame_limit(limit_frames, get_number_of_frames(input_data))
    print("Total frames : ", total_frames)
    allRange = np.arange(total_frames)
    splitRange = np.array_split(allRange, frame_offset)

    slice_list = []
    if resize : 
        filtered_array = numpy.zeros((len(splitRange), resize, resize, 3), numpy.uint8)
    else :
        filtered_array = numpy.zeros((len(splitRange), height, width, 3), numpy.uint8)

    for chunks in splitRange : 
        if resize : 
            median_array = numpy.zeros((len(chunks), resize, resize, 3), numpy.uint8)
        else :
            median_array = numpy.zeros((len(chunks), height, width, 3), numpy.uint8)
        ind = 0
        for frame_number in chunks :
            next_im = get_frame_data(input_data, frame_number)
            if resize : 
                next_im = next_im.resize((resize, resize))
            next_array = numpy.array(next_im, numpy.uint8)
            del next_im
            median_array[ind, :, :, :] = next_array
            ind += 1   
        slice_list.append(median_array)
    results = [median_calc(slice_list[0])]
    for frame in range(len(results)):
        filtered_array[frame, :, :, 0] = results[frame][0]
        filtered_array[frame, :, :, 1] = results[frame][1]
        filtered_array[frame, :, :, 2] = results[frame][2]
        img = Image.fromarray(filtered_array[frame, :, :, :])
        frame_name = frame_path + str(frame) + "." + output_format
        img.save(frame_name, format=output_format)
    del results, filtered_array, median_array
    return frame_path


def median_calc(median_array):
    return numpy.median(median_array[:, :, :, 0], axis=0), \
           numpy.median(median_array[:, :, :, 1], axis=0), \
           numpy.median(median_array[:, :, :, 2], axis=0)


def make_a_video(output_dir, output_format, name):
    if not output_dir.endswith("/"):
        output_dir += "/"
    os.system('ffmpeg -r 24 -i ' + output_dir + '%d.' + output_format + ' -c:v libx264 ' + output_dir + name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._optionals.title = 'arguments'

    parser.add_argument("-i", "--input_file",
                        help="Input csv file which consists of all the videos.", required=True)
    parser.add_argument("--input_dir",
                        help="Input directory for the videos", required=True)
    parser.add_argument("--output_dir",
                        help="Output directory for the Background frames", required=True)
    parser.add_argument("-r", "--resize", default=-1, type=int,
                        help="Resize videos to (r,r)", required=False)
    parser.add_argument("-offset", "--frame_offset", default=1, type=int,
                        help="Keep it 1 for one background frame per video")
    parser.add_argument("-l", "--frame_limit", default=-1, type=int,
                        help="Limit number of frames to specified int (optional)")
    parser.add_argument("-format", "--output_format", default="JPEG", help="Output image format. (optional)")

    args = parser.parse_args()
    dataset_csv = pd.read_csv(args.input_file, header=None)
    if not args.input_dir.endswith("/"):
        args.input_dir += "/"
    if not args.output_dir.endswith("/"):
        args.output_dir += "/"

    print(len(dataset_csv))
    for idx in range(len(dataset_csv)):
        path = args.input_dir + dataset_csv.iloc[idx, 0]
        output_dir = args.output_dir + dataset_csv.iloc[idx, 0] 
        print("------------------------------------------------------------------")
        print("Input path : ", path)
        print("Output path : ", output_dir)

        output_path = temporal_median_filter_multi2(
            make_a_glob(path),
            output_dir,
            args.frame_limit,
            args.output_format,
            args.frame_offset,
            args.resize, 
        )
