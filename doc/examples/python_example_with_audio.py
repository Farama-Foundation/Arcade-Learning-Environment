import os
import sys
import numpy as np
from ale_python_interface import ALEInterface

class Atari:
  def __init__(self, rom_dir):
    self.ale = ALEInterface()

    # Set settings
    self.ale.setInt("random_seed", 123)
    self.frame_skip = 4
    self.ale.setInt("frame_skip", self.frame_skip)
    self.ale.setBool("display_screen", False)
    self.ale.setBool("sound", True)
    self.record_sound_for_user = True
    self.ale.setBool("record_sound_for_user", self.record_sound_for_user) 

    # NOTE recording audio to file still works. But if both file recording and 
    # record_sound_for_user are enabled, then only the latter is done 
    #  self.ale.setString("record_sound_filename", "")

    # Get settings
    self.ale.loadROM(rom_dir)
    self.action_count = 0
    self.screen_width, self.screen_height = self.ale.getScreenDims()
    self.legal_actions = self.ale.getLegalActionSet()
    self.framerate = 60 # Should read from ALE settings technically
    self.samples_per_frame = 512 # Should read from ALE SoundExporter class technically
    self.audio_freq = self.framerate*self.samples_per_frame
    self.all_audio = np.zeros((0,),dtype=np.uint8)

  def take_action(self):
    action = self.legal_actions[np.random.randint(self.legal_actions.size)]
    self.ale.act(action);

  def get_image_and_audio(self):
    np_data_image = np.zeros(self.screen_width*self.screen_height*3, dtype=np.uint8)
    if self.record_sound_for_user:
        np_data_audio = np.zeros(self.ale.getAudioSize(), dtype=np.uint8)
        self.ale.getScreenRGBAndAudio(np_data_image, np_data_audio)

        # Also supports independent audio queries if user desires:
        #  self.ale.getAudio(np_data_audio)
    else:
        np_data_audio = 0
        self.ale.getScreenRGB(np_data_image)

    return np.reshape(np_data_image, (self.screen_height, self.screen_width, 3)), np.asarray(np_data_audio)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print('Usage: %s rom_file' % sys.argv[0])
    sys.exit()

  rom_file = str.encode(sys.argv[1])
  atari = Atari(rom_file)

  while not atari.ale.game_over():
    image, audio = atari.get_image_and_audio()
    atari.take_action()
    atari.action_count += 1
    print '[atari.py] Overall game frame count:', atari.action_count*atari.frame_skip
