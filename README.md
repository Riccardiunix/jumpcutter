# jumpcutter

Automatically edits videos. Explanation (of the origianl fork) here: https://www.youtube.com/watch?v=DQ8orIurGxw

## Some heads-up:

It uses Python 3.

It works on Manjaro 20.0.3 and maybe on Windows 10 (I have to check it)

This program relies heavily on ffmpeg. It will start subprocesses that call ffmpeg, so be aware of that!

As the program runs, it saves every frame of the video as an image file in a
temporary folder. If your video is long, this could take a LOT of space and time.
