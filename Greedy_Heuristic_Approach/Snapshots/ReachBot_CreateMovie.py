import numpy as np
import imageio
import os
import glob

# Inspiration from the following stackoverflow which pushed me in the right direction:
# https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python

path = os.getcwd()

filenames = glob.glob("*.png")
filenames.sort(key=os.path.getmtime)

images = list(map(lambda filename: imageio.imread(filename), filenames))

imageio.mimsave(os.path.join('ReachBot_Solution_Animated.gif'), images, duration = 0.5) # modify duration as needed