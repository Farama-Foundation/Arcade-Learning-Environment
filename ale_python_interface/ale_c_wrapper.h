#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

#include <ale_interface.hpp>

extern "C" {
  // Declares int rgb_palette[256]
  ALEInterface *ALE_new() {return new ALEInterface();}
  void ALE_del(ALEInterface *ale){delete ale;}
  const char *getString(ALEInterface *ale, const char *key){return ale->getString(key).c_str();}
  int getInt(ALEInterface *ale,const char *key) {return ale->getInt(key);}
  bool getBool(ALEInterface *ale,const char *key){return ale->getBool(key);}
  float getFloat(ALEInterface *ale,const char *key){return ale->getFloat(key);}
  void setString(ALEInterface *ale,const char *key,const char *value){ale->setString(key,value);}
  void setInt(ALEInterface *ale,const char *key,int value){ale->setInt(key,value);}
  void setBool(ALEInterface *ale,const char *key,bool value){ale->setBool(key,value);}
  void setFloat(ALEInterface *ale,const char *key,float value){ale->setFloat(key,value);}
  void loadROM(ALEInterface *ale,const char *rom_file){ale->loadROM(rom_file);}
  int act(ALEInterface *ale,int action){return ale->act((Action)action);}
  bool game_over(ALEInterface *ale){return ale->game_over();}
  void reset_game(ALEInterface *ale){ale->reset_game();}
  void getAvailableModes(ALEInterface *ale,int *availableModes) {
    ModeVect modes_vect = ale->getAvailableModes();
    for(unsigned int i = 0; i < ale->getAvailableModes().size(); i++){
      availableModes[i] = modes_vect[i];
    }
  }
  int getAvailableModesSize(ALEInterface *ale) {return ale->getAvailableModes().size();}
  void setMode(ALEInterface *ale, int mode) {ale->setMode(mode);}
  void getAvailableDifficulties(ALEInterface *ale,int *availableDifficulties) {
    DifficultyVect difficulties_vect = ale->getAvailableDifficulties();
    for(unsigned int i = 0; i < ale->getAvailableDifficulties().size(); i++){
      availableDifficulties[i] = difficulties_vect[i];
    }
  }
  int getAvailableDifficultiesSize(ALEInterface *ale) {return ale->getAvailableDifficulties().size();}
  void setDifficulty(ALEInterface *ale, int difficulty) {ale->setDifficulty(difficulty);}
  void getLegalActionSet(ALEInterface *ale,int *actions) {
    ActionVect action_vect = ale->getLegalActionSet();
    for(unsigned int i = 0; i < ale->getLegalActionSet().size(); i++){
      actions[i] = action_vect[i];
    }
  }
  int getLegalActionSize(ALEInterface *ale){return ale->getLegalActionSet().size();}
  void getMinimalActionSet(ALEInterface *ale,int *actions){
    ActionVect action_vect = ale->getMinimalActionSet();
    for(unsigned int i = 0;i < ale->getMinimalActionSet().size();i++){
      actions[i] = action_vect[i];
    }
  }
  int getMinimalActionSize(ALEInterface *ale){return ale->getMinimalActionSet().size();}
  int getFrameNumber(ALEInterface *ale){return ale->getFrameNumber();}
  int lives(ALEInterface *ale){return ale->lives();}
  int getEpisodeFrameNumber(ALEInterface *ale){return ale->getEpisodeFrameNumber();}
  void getScreen(ALEInterface *ale,unsigned char *screen_data){
    int w = ale->getScreen().width();
    int h = ale->getScreen().height();
    pixel_t *ale_screen_data = (pixel_t *)ale->getScreen().getArray();
    memcpy(screen_data,ale_screen_data,w*h*sizeof(pixel_t));
  }
  void getRAM(ALEInterface *ale,unsigned char *ram){
    unsigned char *ale_ram = ale->getRAM().array();
    int size = ale->getRAM().size();
    memcpy(ram,ale_ram,size*sizeof(unsigned char));
  }
  int getRAMSize(ALEInterface *ale){return ale->getRAM().size();}
  int getScreenWidth(ALEInterface *ale){return ale->getScreen().width();}
  int getScreenHeight(ALEInterface *ale){return ale->getScreen().height();}

  void getScreenRGB(ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteRGB(output_buffer, ale_screen_data, screen_size );
  }

  void getScreenGrayscale(ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteGrayscale(output_buffer, ale_screen_data, screen_size);
  }

  void saveState(ALEInterface *ale){ale->saveState();}
  void loadState(ALEInterface *ale){ale->loadState();}
  ALEState* cloneState(ALEInterface *ale){return new ALEState(ale->cloneState());}
  void restoreState(ALEInterface *ale, ALEState* state){ale->restoreState(*state);}
  ALEState* cloneSystemState(ALEInterface *ale){return new ALEState(ale->cloneSystemState());}
  void restoreSystemState(ALEInterface *ale, ALEState* state){ale->restoreSystemState(*state);}
  void deleteState(ALEState* state){delete state;}
  void saveScreenPNG(ALEInterface *ale,const char *filename){ale->saveScreenPNG(filename);}

  // Encodes the state as a raw bytestream. This may have multiple '\0' characters
  // and thus should not be treated as a C string. Use encodeStateLen to find the length
  // of the buffer to pass in, or it will be overrun as this simply memcpys bytes into the buffer.
  void encodeState(ALEState *state, char *buf, int buf_len);
  int encodeStateLen(ALEState *state);
  ALEState *decodeState(const char *serialized, int len);

  // 0: Info, 1: Warning, 2: Error
  void setLoggerMode(int mode) { ale::Logger::setMode(ale::Logger::mode(mode)); }
}

#endif
