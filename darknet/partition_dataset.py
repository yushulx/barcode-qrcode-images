""" usage: partition_dataset.py [-h] [-i IMAGEDIR] [-r RATIO]

Partition dataset of images into training and testing sets

optional arguments:
  -h, --help            show this help message and exit
  -i IMAGEDIR, --imageDir IMAGEDIR
                        Path to the folder where the image dataset is stored. If not specified, the CWD will be used.
                        Path to the output folder where the train and test dirs should be created. Defaults to the same directory as IMAGEDIR.
  -r RATIO, --ratio RATIO
                        The ratio of the number of test images over the total number of images. The default is 0.1.
"""
import os
import re
from shutil import copyfile
import argparse
import math
import random


def iterate_dir(source, ratio):
    source = source.replace('\\', '/')
    train_dir = 'data/obj/train'
    test_dir = 'data/obj/test'

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    image_files = []

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        image_files.append("data/obj/test/" + filename)
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        txt_filename = os.path.splitext(filename)[0]+'.txt'
        copyfile(os.path.join(source, txt_filename),
                 os.path.join(test_dir, txt_filename))

        images.remove(images[idx])

    with open("data/test.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()

    image_files = []

    for filename in images:
        image_files.append("data/obj/train/" + filename)
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        txt_filename = os.path.splitext(filename)[0]+'.txt'
        copyfile(os.path.join(source, txt_filename),
                 os.path.join(train_dir, txt_filename))

    with open("data/train.txt", "w") as outfile:
        for image in image_files:
            outfile.write(image)
            outfile.write("\n")
        outfile.close()


def main():

    # Initiate argument parser
    parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--imageDir',
        help='Path to the folder where the image dataset is stored. If not specified, the CWD will be used.',
        type=str,
        default=os.getcwd()
    )
    parser.add_argument(
        '-r', '--ratio',
        help='The ratio of the number of test images over the total number of images. The default is 0.1.',
        default=0.1,
        type=float)
    args = parser.parse_args()

    # Now we are ready to start the iteration
    iterate_dir(args.imageDir, args.ratio)


if __name__ == '__main__':
    main()
