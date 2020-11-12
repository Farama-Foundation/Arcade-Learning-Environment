#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

#include <ale_interface.hpp>

extern "C" {
  // Declares int rgb_palette[256]
  ale::ALEInterface *ALE_new() {return new ale::ALEInterface();}
  void ALE_del(ale::ALEInterface *ale){delete ale;}
  const char *getString(ale::ALEInterface *ale, const char *key){return ale->getString(key).c_str();}
  int getInt(ale::ALEInterface *ale,const char *key) {return ale->getInt(key);}
  bool getBool(ale::ALEInterface *ale,const char *key){return ale->getBool(key);}
  float getFloat(ale::ALEInterface *ale,const char *key){return ale->getFloat(key);}
  void setString(ale::ALEInterface *ale,const char *key,const char *value){ale->setString(key,value);}
  void setInt(ale::ALEInterface *ale,const char *key,int value){ale->setInt(key,value);}
  void setBool(ale::ALEInterface *ale,const char *key,bool value){ale->setBool(key,value);}
  void setFloat(ale::ALEInterface *ale,const char *key,float value){ale->setFloat(key,value);}
  void loadROM(ale::ALEInterface *ale,const char *rom_file){ale->loadROM(rom_file);}
  int isSupportedRom(ale::ALEInterface *ale){return ale->isSupportedRom();}
  int act(ale::ALEInterface *ale,int action){return ale->act((ale::Action)action);}
  int actMultiSize(ale::ALEInterface *ale){return ale->numPlayersActive();}
  void actMulti(ale::ALEInterface *ale,int *actions,int * rewards) {
    int num_players = ale->numPlayersActive();
    std::vector<ale::Action> actions_v(num_players);
    for(int i = 0; i < num_players; i++){
      actions_v[i] = (ale::Action)actions[i];
    }
    std::vector<ale::reward_t> rewards_v = ale->act(actions_v);
    for(int i = 0; i < num_players; i++){
      rewards[i] = rewards_v[i];
    }
  }
  bool game_over(ale::ALEInterface *ale){return ale->game_over();}
  void reset_game(ale::ALEInterface *ale){ale->reset_game();}
  void getAvailableModes(ale::ALEInterface *ale,int *availableModes,int num_players) {
    ale::ModeVect modes_vect = ale->getAvailableModes(num_players);
    for(unsigned int i = 0; i < modes_vect.size(); i++){
      availableModes[i] = modes_vect[i];
    }
  }
  int getAvailableModesSize(ale::ALEInterface *ale,int num_players) {return ale->getAvailableModes(num_players).size();}
  void setMode(ale::ALEInterface *ale, int mode) {ale->setMode(mode);}
  void getAvailableDifficulties(ale::ALEInterface *ale,int *availableDifficulties) {
    ale::DifficultyVect difficulties_vect = ale->getAvailableDifficulties();
    for(unsigned int i = 0; i < ale->getAvailableDifficulties().size(); i++){
      availableDifficulties[i] = difficulties_vect[i];
    }
  }
  int getAvailableDifficultiesSize(ale::ALEInterface *ale) {return ale->getAvailableDifficulties().size();}
  void setDifficulty(ale::ALEInterface *ale, int difficulty) {ale->setDifficulty(difficulty);}
  void getLegalActionSet(ale::ALEInterface *ale,int *actions) {
    ale::ActionVect action_vect = ale->getLegalActionSet();
    for(unsigned int i = 0; i < ale->getLegalActionSet().size(); i++){
      actions[i] = action_vect[i];
    }
  }
  int getLegalActionSize(ale::ALEInterface *ale){return ale->getLegalActionSet().size();}
  void getMinimalActionSet(ale::ALEInterface *ale,int *actions){
    ale::ActionVect action_vect = ale->getMinimalActionSet();
    for(unsigned int i = 0;i < ale->getMinimalActionSet().size();i++){
      actions[i] = action_vect[i];
    }
  }
  int getMinimalActionSize(ale::ALEInterface *ale){return ale->getMinimalActionSet().size();}
  int getFrameNumber(ale::ALEInterface *ale){return ale->getFrameNumber();}
  void lives(ale::ALEInterface *ale,int *lives){
    auto live_vect = ale->allLives();
    for(unsigned int i = 0;i < ale->allLives().size();i++){
      lives[i] = live_vect[i];
    }
  }
  int livesSize(ale::ALEInterface *ale){return ale->allLives().size();}
  int getEpisodeFrameNumber(ale::ALEInterface *ale){return ale->getEpisodeFrameNumber();}
  void getScreen(ale::ALEInterface *ale,unsigned char *screen_data){
    int w = ale->getScreen().width();
    int h = ale->getScreen().height();
    ale::pixel_t *ale_screen_data = (ale::pixel_t *)ale->getScreen().getArray();
    std::memcpy(screen_data,ale_screen_data,w*h*sizeof(ale::pixel_t));
  }
  void getRAM(ale::ALEInterface *ale,unsigned char *ram){
    const unsigned char *ale_ram = ale->getRAM().array();
    int size = ale->getRAM().size();
    std::memcpy(ram,ale_ram,size*sizeof(unsigned char));
  }
  void setRAM(ale::ALEInterface *ale, size_t memory_index, ale::byte_t value) {
    return ale->setRAM(memory_index, value);
  }
  int getRAMSize(ale::ALEInterface *ale){return ale->getRAM().size();}
  int getScreenWidth(ale::ALEInterface *ale){return ale->getScreen().width();}
  int getScreenHeight(ale::ALEInterface *ale){return ale->getScreen().height();}

  void getScreenRGB(ale::ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    ale::pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteRGB(output_buffer, ale_screen_data, screen_size );
  }

  void getScreenGrayscale(ale::ALEInterface *ale, unsigned char *output_buffer){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;
    ale::pixel_t *ale_screen_data = ale->getScreen().getArray();

    ale->theOSystem->colourPalette().applyPaletteGrayscale(output_buffer, ale_screen_data, screen_size);
  }

  void saveState(ale::ALEInterface *ale){ale->saveState();}
  void loadState(ale::ALEInterface *ale){ale->loadState();}
  ale::ALEState* cloneState(ale::ALEInterface *ale){return new ale::ALEState(ale->cloneState());}
  void restoreState(ale::ALEInterface *ale, ale::ALEState* state){ale->restoreState(*state);}
  ale::ALEState* cloneSystemState(ale::ALEInterface *ale){return new ale::ALEState(ale->cloneSystemState());}
  void restoreSystemState(ale::ALEInterface *ale, ale::ALEState* state){ale->restoreSystemState(*state);}
  void deleteState(ale::ALEState* state){delete state;}
  void saveScreenPNG(ale::ALEInterface *ale,const char *filename){ale->saveScreenPNG(filename);}

  // Encodes the state as a raw bytestream. This may have multiple '\0' characters
  // and thus should not be treated as a C string. Use encodeStateLen to find the length
  // of the buffer to pass in, or it will be overrun as this simply memcpys bytes into the buffer.
  void encodeState(ale::ALEState *state, char *buf, int buf_len);
  int encodeStateLen(ale::ALEState *state);
  ale::ALEState *decodeState(const char *serialized, int len);

  // 0: Info, 1: Warning, 2: Error
  void setLoggerMode(int mode) { ale::Logger::setMode(ale::Logger::mode(mode)); }
}

#endif
