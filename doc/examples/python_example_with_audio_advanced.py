import sys
import os
import shutil
import numpy as np
import subprocess as sp
from ale_python_interface import ALEInterface
import scipy.io.wavfile as wavfile
import scipy.misc
from python_speech_features import mfcc
import matplotlib.pyplot as plt

class Atari:
  def __init__(self, rom_dir):
    self.ale = ALEInterface()

    # Set settings
    self.ale.setInt("random_seed", 123)
    self.frame_skip = 1
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
    self.audio_freq = self.framerate*self.samples_per_frame#/self.frame_skip
    self.all_audio = np.zeros((0,),dtype=np.uint8)

    # Saving audio/video to disk for verification. 
    self.save_to_file = True # NOTE set to False to test actual screen/audio query speed!
    if self.save_to_file:
        self.save_dir_av = './logs_av_seq_Example' # Save png sequence and audio wav file here
        self.save_dir_movies = './log_movies_Example'
        self.save_image_prefix = 'image_frames'
        self.save_audio_filename = 'audio_user_recorder.wav'
        self.create_save_dir(self.save_dir_av)

  def take_action(self):
    action = self.legal_actions[np.random.randint(self.legal_actions.size)]
    self.ale.act(action);

  def create_save_dir(self, directory):
    # Remove previous img/audio image logs
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

  def get_image_and_audio(self):
    np_data_image = np.zeros(self.screen_width*self.screen_height*3, dtype=np.uint8)
    if self.record_sound_for_user:
        np_data_audio = np.zeros(self.ale.getAudioSize(), dtype=np.uint8)
        self.ale.getScreenRGBAndAudio(np_data_image, np_data_audio)

        # Also supports independent audio queries if user desires:
        #  self.ale.getAudio(np_data_audio)
    else:
        #  np_data_audio = 0
        np_data_audio = np.zeros(self.ale.getAudioSize(), dtype=np.uint8)
        self.ale.getAudio(np_data_audio)
        self.ale.getScreenRGB(np_data_image)

    return np.reshape(np_data_image, (self.screen_height, self.screen_width, 3)), np.asarray(np_data_audio)

  def audio_to_mfcc(self, audio):
    mfcc_data = mfcc(signal=audio, samplerate=self.audio_freq, winlen=0.002, winstep=0.0006)
    mfcc_data = np.swapaxes(mfcc_data, 0 ,1) # Time on x-axis

    # Normalization 
    min_data = np.min(mfcc_data.flatten())
    max_data = np.max(mfcc_data.flatten())
    mfcc_data = (mfcc_data-min_data)/(max_data-min_data)
     
    return mfcc_data

  def save_image(self, image):
    number = str(self.action_count).zfill(6)
    scipy.misc.imsave(os.path.join(self.save_dir_av, self.save_image_prefix+number+'.png'), image)

  def save_audio(self, audio):
    wavfile.write(os.path.join(self.save_dir_av, self.save_audio_filename), self.audio_freq, audio)

  def save_movie(self, movie_name):
    # Use ffmpeg to convert the saved img sequences and audio to mp4

    # Video recording
    command = [ "ffmpeg",
                '-y', # overwrite output file if it exists
                '-r', str(self.framerate), # frames per second
                '-i', os.path.join(self.save_dir_av, self.save_image_prefix+'%6d.png') # Video input comes from pngs
              ]

    # Audio if available
    if self.record_sound_for_user:
        command.extend(['-i', os.path.join(self.save_dir_av, self.save_audio_filename)]) # Audio input comes from wav

    # Codecs and output
    command.extend(['-c:v', 'libx264', # Video codec
                '-c:a', 'mp3', # Audio codec
                os.path.join(self.save_dir_movies, movie_name+'.mp4') # Output dir
                   ])

    # Make movie dir and write the mp4
    if not os.path.exists(self.save_dir_movies):
        os.makedirs(self.save_dir_movies)
    sp.call(command) # NOTE: needs ffmpeg! Will throw 'dir doesn't exist err' otherwise.
  
  def concat_image_audio(self, image, audio_mfcc):
    # Concatenates image and audio to test sync'ing in saved .mp4
    audio_mfcc = scipy.misc.imresize(audio_mfcc, np.shape(image)) # Resize MFCC image to be same size as screen image
    cmap = plt.get_cmap('viridis') # Apply a colormap to spectrogram
    audio_mfcc = (np.delete(cmap(audio_mfcc), 3, 2)*255.).astype(np.uint8) # Gray MFCC -> 4 channel colormap -> 3 channel colormap
    image = np.concatenate((image, audio_mfcc), axis=1) # Concat screen image and MFCC image
    return image

  def plot_mfcc(self, audio_mfcc):
    plt.clf()
    plt.imshow(audio_mfcc, interpolation='bilinear', cmap=plt.get_cmap('viridis'))
    plt.pause(0.001)

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print('Usage: %s roms directory' % sys.argv[0])
    sys.exit()

  dir_games = str.encode(sys.argv[1])
  game_names = [os.path.splitext(f)[0] for f in os.listdir(dir_games) if f.endswith('.bin')] # Tests all ROMs

  # Go through all games in dir_games and record audio/video to file
  for game_name in game_names:
      atari = Atari(os.path.join(dir_games, game_name+'.bin'))

      # Run a few hundred frames of each game for verification
      while atari.action_count < 500 and not atari.ale.game_over():
        image, audio = atari.get_image_and_audio()

        # Compute audio spectrogram and save
        if atari.record_sound_for_user and atari.action_count > 0:
            atari.all_audio = np.append(atari.all_audio, audio)
            if atari.save_to_file: 
                audio_mfcc = atari.audio_to_mfcc(audio) # Spectrogram 
                image = atari.concat_image_audio(image, audio_mfcc) # Concat (for visualization only)

        # Save frame (may include picture of audio mfcc, if audio enabled)
        if atari.save_to_file and atari.action_count > 0: 
            atari.save_image(image)

        print '[atari.py] overall game frame count:', atari.action_count*atari.frame_skip

        atari.take_action()
        atari.action_count += 1

      # Convert png/audio sequence to mp4 to test syncing
      if atari.save_to_file: 
          if atari.record_sound_for_user: 
              atari.save_audio(atari.all_audio)
          atari.save_movie(game_name)
