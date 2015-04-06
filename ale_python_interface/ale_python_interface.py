# ale_python_interface.py
# Author: Ben Goodrich
# This directly implements a python version of the arcade learning environment interface.
# It requires the C wrapper library to be built and on shared object path, as "ale_c_wrapper.so"

from ctypes import cdll
import numpy as np
from numpy.ctypeslib import as_ctypes
import os

ale_lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__),
                                        'libale_c.so'))

class ALEInterface(object):
    def __init__(self):
        self.obj = ale_lib.ALE_new()

    def getString(self, key):
        return ale_lib.getString(self.obj, key)
    def getInt(self, key):
        return ale_lib.getInt(self.obj, key)
    def getBool(self, key):
        return ale_lib.getBool(self.obj, key)
    def getFloat(self, key):
        return ale_lib.getFloat(self.obj, key)

    def setString(self, key, value):
      ale_lib.setString(self.obj, key, value)
    def setInt(self, key, value):
      ale_lib.setInt(self.obj, key, value)
    def setBool(self, key, value):
      ale_lib.setBool(self.obj, key, value)
    def setFloat(self, key, value):
      ale_lib.setFloat(self.obj, key, value)

    def loadROM(self, rom_file):
        ale_lib.loadROM(self.obj, rom_file)

    def act(self, action):
        return ale_lib.act(self.obj, int(action))

    def game_over(self):
        return ale_lib.game_over(self.obj)

    def reset_game(self):
        ale_lib.reset_game(self.obj)

    def getLegalActionSet(self):
        act_size = ale_lib.getLegalActionSize(self.obj)
        act = np.zeros((act_size), dtype=np.int32)
        ale_lib.getLegalActionSet(self.obj, as_ctypes(act))
        return act

    def getMinimalActionSet(self):
        act_size = ale_lib.getMinimalActionSize(self.obj)
        act = np.zeros((act_size), dtype=np.int32)
        ale_lib.getMinimalActionSet(self.obj, as_ctypes(act))
        return act

    def getFrameNumber(self):
        return ale_lib.getFrameNumber(self.obj)

    def lives(self):
        return ale_lib.lives(self.obj)

    def getEpisodeFrameNumber(self):
        return ale_lib.getEpisodeFrameNumber(self.obj)

    def getScreenDims(self):
        """returns a tuple that contains (screen_width, screen_height)
        """
        width = ale_lib.getScreenWidth(self.obj)
        height = ale_lib.getScreenHeight(self.obj)
        return (width, height)


    def getScreen(self, screen_data=None):
        """This function fills screen_data with the RAW Pixel data
        screen_data MUST be a numpy array of uint8/int8. This could be initialized like so:
        screen_data = np.array(w*h, dtype=np.uint8)
        Notice,  it must be width*height in size also
        If it is None,  then this function will initialize it
        Note: This is the raw pixel values from the atari,  before any RGB palette transformation takes place
        """
        if(screen_data is None):
            width = ale_lib.getScreenWidth(self.obj)
            height = ale_lib.getScreenWidth(self.obj)
            screen_data = np.zeros(width*height, dtype=np.uint8)
        ale_lib.getScreen(self.obj, as_ctypes(screen_data))
        return screen_data

    def getScreenRGB(self, screen_data=None):
        """This function fills screen_data with the data
        screen_data MUST be a numpy array of uint32/int32. This can be initialized like so:
        screen_data = np.array(w*h, dtype=np.uint32)
        Notice,  it must be width*height in size also
        If it is None,  then this function will initialize it
        """
        if(screen_data is None):
            width = ale_lib.getScreenWidth(self.obj)
            height = ale_lib.getScreenWidth(self.obj)
            screen_data = np.zeros(width*height, dtype=np.uint32)
        ale_lib.getScreenRGB(self.obj, as_ctypes(screen_data))
        return screen_data

    def getRAMSize(self):
        return ale_lib.getRAMSize(self.obj)

    def getRAM(self, ram=None):
        """This function grabs the atari RAM.
        ram MUST be a numpy array of uint8/int8. This can be initialized like so:
        ram = np.array(ram_size, dtype=uint8)
        Notice: It must be ram_size where ram_size can be retrieved via the getRAMSize function.
        If it is None,  then this function will initialize it.
        """
        if(ram is None):
            ram_size = ale_lib.getRAMSize(self.obj)
            ram = np.zeros(ram_size, dtype=np.uint8)
        ale_lib.getRAM(self.obj, as_ctypes(ram))

    def saveScreenPNG(self, filename):
        return ale_lib.saveScreenPNG(self.obj, filename)

    def __del__(self):
        ale_lib.ALE_del(self.obj)
