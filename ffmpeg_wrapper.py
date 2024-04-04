import subprocess
from PIL import Image
from glob import glob 
import os
import argparse


def ffmpeg_wrapper(dirname, framerate=25, output_file="output.mp4"):

    # Get all the images in the directory
    images = sorted(glob(os.path.join(dirname, "*.png")))
    image_width, image_height = Image.open(images[0]).size

    if image_width % 2 != 0:
        image_width = image_width - 1
    if image_height % 2 != 0:
        image_height = image_height - 1

    if os.path.dirname(output_file) == "":
        output_file = os.path.join(dirname, output_file)

    command = ["ffmpeg", 
               "-r", str(framerate),
               "-f", "image2",
               "-i", os.path.join(dirname, "%*.png"),
               "-filter:v", f"crop={image_width}:{image_height}",
               "-vcodec", "libx264",
                "-crf", "25",
               "-pix_fmt", "yuv420p", 
               output_file]
    subprocess.run(command)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a directory of images to a video using ffmpeg")
    parser.add_argument("dirname", type=str, help="Directory containing images")
    parser.add_argument("-r","--framerate", type=int, default=25, help="Framerate of the output video")
    parser.add_argument("-o","--output_file", type=str, default="output.mp4", help="Name of the output file")
    args = parser.parse_args()
    ffmpeg_wrapper(args.dirname, args.framerate, args.output_file)