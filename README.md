# cam_data_conversion
A python script using OpenCV, PIL, and ufp packages to convert Raspberry Pi Camera feed into SWARM Lab Research Camera quality feed

This script takes a RPi video input file and converts it real-time to SWARM Lab Research camera quality video file output. This involves a reduction in image resolution, bits per pixel, greyscale and reduced field of view.
The output video encoding is hardware specific, functional on Mac OSX Sierra.
To run conversion on a video file, set the media_path and media_name in the # SCRIPT # of cam_data_conversion.py, and then run the python script.
The image_utils.py file is a modified python script of the ufp package to be python 3 compatible.

TODO:
- Connect to RPi camera on Donkey Car to convert and save live-captured video for future training purposes.

https://docs.google.com/presentation/d/1hXQC8OLIvuCYonEwlCy9TCExW7WrS5Bjrg0a-a1qPoA/edit?usp=sharing