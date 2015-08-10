#ifndef __ALE_C_WRAPPER_H__
#define __ALE_C_WRAPPER_H__

#include <ale_interface.hpp>

extern "C" {
  // Declares int rgb_palette[256]
#include "atari_ntsc_rgb_palette.h"
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
  void getLegalActionSet(ALEInterface *ale,int *actions){
    ActionVect action_vect = ale->getLegalActionSet();
    for(unsigned int i = 0;i < ale->getLegalActionSet().size();i++){
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

  void getScreenPalette24Bits(ALEInterface *ale, unsigned char *screen_data, const int palette[]){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;

    pixel_t *ale_screen_data = ale->getScreen().getArray();
    pixel_t *p = ale_screen_data;
    pixel_t *q = screen_data;

    for(size_t i = 0; i < screen_size; i++, p++){
      int rgb = palette[*p];
      *q = (unsigned char) ((rgb >> 16));  q++;    // r
      *q = (unsigned char) ((rgb >>  8));  q++;    // g
      *q = (unsigned char) ((rgb >>  0));  q++;    // b
    }
  }

  void getScreenPalette8Bits(ALEInterface *ale, unsigned char *screen_data, const unsigned char palette[]){
    size_t w = ale->getScreen().width();
    size_t h = ale->getScreen().height();
    size_t screen_size = w*h;

    pixel_t *ale_screen_data = ale->getScreen().getArray();
    pixel_t *p = ale_screen_data;
    pixel_t *q = screen_data;

    for(size_t i = 0; i < screen_size; i++, p++, q++){
      *q = palette[*p];
    }
  }

  void getPaletteRGB(ALEInterface *ale, int *palette_data) {
    memcpy(palette_data, &rgb_palette, sizeof(rgb_palette));
  }

  void getScreenRGB(ALEInterface *ale, unsigned char *screen_data){
    getScreenPalette24Bits(ale, screen_data, rgb_palette);
  }

  void saveState(ALEInterface *ale){ale->saveState();}
  void loadState(ALEInterface *ale){ale->loadState();}
  void saveScreenPNG(ALEInterface *ale,const char *filename){ale->saveScreenPNG(filename);}
}

#endif
