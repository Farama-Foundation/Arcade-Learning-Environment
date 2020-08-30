#include "ale_interface.hpp"
#include <fstream>
#include <vector>

using namespace ale;
using namespace std;

void save_frame(std::vector<unsigned char> & Buff);

constexpr int steps_to_test = 2000;

std::vector<std::string> two_player_games;
void init_two_player_fnames(){
  std::vector<std::string> two_player_fnames = {
    "video_checkers", // this actually passes the tests, the test just doesn't play p1 well by default
    "tennis",
    "othello", //othello seems to be working, just isn't passing the test
    "double_dunk",
    "boxing",
    "combat",
    "entombed",
    "fishing_derby",
    "flag_capture",
    "ice_hockey",
    "joust",
    "mario_bros",
    "maze_craze",
    "pong",
    "space_invaders",
    "space_war",
    "surround",
    "wizard_of_wor",
    "warlords",
  };
  std::string main_path = "roms/";//"/home/benblack/anaconda3/lib/python3.7/site-packages/ale_py/ROM/";
  for(std::string fname : two_player_fnames){
    std::string new_fname = main_path + fname + ".bin";
    two_player_games.push_back(new_fname);
    //std::ifstream  src(new_fname, std::ios::binary);
    //std::ofstream  dst("roms/"+fname+".bin",   std::ios::binary);

  //  dst << src.rdbuf();
  }
}
bool test_env_exists(std::string fname){
  ifstream file(fname);
  if(!file){
    return false;
  }
  file.close();
  ALEInterface interface;

  interface.loadROM(fname);
  return true;
}
bool test_two_player(std::string fname){
  ALEInterface interface;

  interface.loadROM(fname);
  interface.reset_game();
  return interface.supportsNumPlayers(2) ;
}
bool test_four_player(std::string fname){
  ALEInterface interface;

  interface.loadROM(fname);
  interface.reset_game();
  return interface.supportsNumPlayers(4) ;
}
std::size_t hash_vec(std::vector<uint8_t> const& vec) {
  std::size_t seed = vec.size();
  for(uint8_t i : vec) {
    seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
std::size_t hash_together(std::size_t h1, std::size_t h2){
  return h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
}
size_t play_sequence(ALEInterface & interface,int sequence_num){
  size_t hashCode = 0;
  std::vector<unsigned char> output_rgb_buffer(192*160*3);
  //save_frame(output_rgb_buffer);
  ActionVect min_actionsp1 = interface.getMinimalActionSet();
  int action_p1 = 0;
  for(int j = 0; j < steps_to_test; j++){
    if((sequence_num == 0 && j % 4 == 0) || (sequence_num == 1 && j % 64 == 0)){
       action_p1 = (std::hash<int>()(j))%min_actionsp1.size();
    }

    interface.act(min_actionsp1[action_p1]);

    interface.getScreenRGB(output_rgb_buffer);
    hashCode = hash_together(hashCode,hash_vec(output_rgb_buffer));

    if(j % 43 == 0){
    //  save_frame(output_rgb_buffer);
    }
  }
  return hashCode;
}
size_t play_sequence_p2(ALEInterface & interface,int sequence_num){
  size_t hashCode = 0;
  std::vector<unsigned char> output_rgb_buffer(192*160*3);
  //save_frame(output_rgb_buffer);
  ActionVect min_actionsp1 = interface.getMinimalActionSet();
  int action_p2 = 0;
  int action_p1 = 0;
  for(int j = 0; j < steps_to_test; j++){
    if(j % 32 == 0){
       action_p1 = (std::hash<int>()(j))%min_actionsp1.size();
    }
    if((sequence_num == 0 && j % 4 == 0) || (sequence_num == 1 && j % 64 == 0)){
      action_p2 = std::hash<int>()(j)%min_actionsp1.size();
    }

    interface.act({min_actionsp1[action_p1], min_actionsp1[action_p2]});

    interface.getScreenRGB(output_rgb_buffer);
    hashCode = hash_together(hashCode,hash_vec(output_rgb_buffer));
    if(interface.game_over()){
      std::cout << "game ended early\n";
      break;
    }
    if(j % 14 == 0){
    //  save_frame(output_rgb_buffer);
    }
  }
  return hashCode;
}
size_t play_sequence_p4(ALEInterface & interface,int sequence_num){
  size_t hashCode = 0;
  std::vector<unsigned char> output_rgb_buffer(192*160*3);
  //save_frame(output_rgb_buffer);
  ActionVect min_actionsp1 = interface.getMinimalActionSet();
  int action_p4 = 0;
  int action_p1 = 0;
  for(int j = 0; j < steps_to_test; j++){
    if(j % 8 == 0){
       action_p1 = (std::hash<int>()(j))%min_actionsp1.size();
    }
    if((sequence_num == 0 && j % 4 == 0) || (sequence_num == 1 && j % 64 == 0)){
       action_p4 = (std::hash<int>()(j))%min_actionsp1.size();
    }

    interface.act(std::vector<Action>{min_actionsp1[action_p1], min_actionsp1[action_p1], min_actionsp1[action_p1], min_actionsp1[action_p4]});

    interface.getScreenRGB(output_rgb_buffer);
    hashCode = hash_together(hashCode,hash_vec(output_rgb_buffer));
    if(interface.game_over()){
      std::cout << "game ended early\n";
      break;
    }
    if(j % 216 == 0){
     //save_frame(output_rgb_buffer);
    }
  }
  return hashCode;
}
bool test_two_player_controlability(std::string fname){
  int seed = 123982;
  size_t hashs[2] = {0,0};
  for(int i = 0; i < 2; i++){
    srand(seed);
    ALEInterface interface;
    interface.setInt("random_seed", seed);
    interface.loadROM(fname);
    ModeVect modes = interface.getAvailableModes(2);
    interface.setMode(modes[0]);
    interface.reset_game();
    hashs[i] = play_sequence_p2(interface,i);
  }
  return hashs[0] != hashs[1];
}
bool test_four_player_controlability(std::string fname){
  int seed = 123982;
  size_t hashs[2] = {0,0};
  std::cout << "running 4p test\n";
  for(int i = 0; i < 2; i++){
    srand(seed);
    ALEInterface interface;
    interface.setInt("random_seed", seed);
    interface.loadROM(fname);
    ModeVect modes = interface.getAvailableModes(4);
    interface.setMode(modes[0]);
    interface.reset_game();
    hashs[i] = play_sequence_p4(interface,i);
  }
  return hashs[0] != hashs[1];
}
bool test_single_player_controlability(std::string fname){
  int seed = 123982;
  size_t hashs[2] = {0,0};
  for(int i = 0; i < 2; i++){
    srand(seed);
    ALEInterface interface;
    interface.setInt("random_seed", seed);
    interface.loadROM(fname);
    ModeVect modes = interface.getAvailableModes();
    interface.setMode(modes[modes.size()-1]);
    interface.reset_game();
    hashs[i] = play_sequence(interface,i);
  }
  return hashs[0] != hashs[1];
}
void test_all_default(){
  using namespace std;
  ifstream file("md5.txt");
  assert(file);
  vector<string> names;
  vector<string> md5s;
  while(file){
    string name,md5;
    file >> md5;
    file >> name;
    if(name.size() <= 4){
      break;
    }
    string stripname(name.begin(),name.end()-4);
    names.push_back(stripname);
    md5s.push_back(md5);
  }
  for (int i = 0; i < names.size(); i++){
    cout << names[i] << ": worked\n";
    string path_base =  "/home/ben/.virtualenvs/zoo/lib/python3.6/site-packages/ale_py/ROM/";
    string path = path_base + names[i] + "/" + names[i] + ".bin";

    ALEInterface interface;
    interface.loadROM(path);
    interface.reset_game();
    assert(interface.isSupportedRom());
  }

}
int main(){
  //test_all_default();
  init_two_player_fnames();
  for(std::string fname : two_player_games){
    size_t last_slash = fname.find_last_of('/');
    std::string binname(fname.begin() + last_slash, fname.end());
    if(!test_env_exists(fname)){
        cout << "failed to load file!\n";
        cout << fname << "\n";
        exit(-1);
    }
    // else if(!test_two_player(fname)){
    //   cout << binname << " environment bad\n";
    // }
    // else if(true){
    //   cout << binname << " environment not crashing (not necessarily controllable)!\n";
    // }
    else if(test_four_player(fname)){
      test_two_player_controlability(fname);
      if (!test_four_player_controlability(fname) ){
      cout << binname << " environment not controllable in four player mode\n";
    }
    else{
      cout << binname << " environment works in 4player!\n";
    }
    }
    else if(test_two_player(fname)){
      if(!test_two_player_controlability(fname)){
        cout << binname << " environment not controllable in two player mode\n";
      }
      else{
        cout << binname << " environment works in 2player!\n";
      }
    }
    else if(!test_two_player(fname)){
      if(!test_single_player_controlability(fname)){
        cout << binname << " environment not two player and not controllable in single player mode\n";
      }
      else{
        cout << binname << " environment not two player but works in single player\n";
      }
    }
    else{
      cout << binname << " environment passes!\n";
    }
  }
  ALEInterface interface;
  string fname = "/home/benblack/anaconda3/lib/python3.7/site-packages/ale_py/ROM/double_dunk/double_dunk.bin";
  ifstream file(fname);
  if(!file){
    cout << "failed to load file!\n";
    return -1;
  }
  file.close();
  interface.loadROM(fname);
  ModeVect modes = interface.getAvailableModes();
  interface.setMode(modes[modes.size()-1]);
  ActionVect actions = interface.getMinimalActionSet();
  reward_t rew = interface.act(actions[0]);
  std::cout << "reward is: " << rew << "\n";
}

void save_frame_data(FILE *file, std::vector<unsigned char> & Buff);
void save_buffer_header(FILE *file);
constexpr int X_WIN_SIZE = 160;
constexpr int Y_WIN_SIZE = 192;
int frame_count = 0;
void save_frame( std::vector<unsigned char> & Buff){
    //string bmp_name = "tmp/frame.bmp";
    string fcount = to_string(frame_count);
    string zeros = string(6-fcount.size(),'0');
    string bmp_name = "frames/frame"+zeros+fcount+".bmp";
    FILE * file = fopen(bmp_name.c_str(), "w");
    save_buffer_header(file);
    save_frame_data(file,Buff);
    fclose(file);
    //system((" C:/Windows/System32/bash.exe scripts/to_png.sh "+bmp_name+" "+png_name).c_str());
    frame_count++;
}
void save_frame_data(FILE *file, std::vector<unsigned char> & Buff){
    fwrite(Buff.data(),Buff.size(), 1, file);	/* write bmp pixels */
}
using WORD=uint16_t;
using DWORD=uint32_t;
using LONG=uint32_t;

#pragma pack(2)
struct BITMAPFILEHEADER {
  WORD bfType;
  DWORD bfSize;
  WORD bfReserved1;
  WORD bfReserved2;
  DWORD bfOffBits;
};
struct BITMAPINFOHEADER {
  DWORD biSize;
  LONG biWidth;
  LONG biHeight;
  WORD biPlanes;
  WORD biBitCount;
  DWORD biCompression;
  DWORD biSizeImage;
  LONG biXPelsPerMeter;
  LONG biYPelsPerMeter;
  DWORD biClrUsed;
  DWORD biClrImportant;
};
constexpr LONG BI_RGB = 0;
void save_buffer_header(FILE *file){
    BITMAPFILEHEADER bitmapFileHeader;
    BITMAPINFOHEADER bitmapInfoHeader;

    bitmapFileHeader.bfType = 0x4D42;
    bitmapFileHeader.bfSize = sizeof(BITMAPINFOHEADER) + sizeof(BITMAPFILEHEADER)+ X_WIN_SIZE*Y_WIN_SIZE * 3;
    bitmapFileHeader.bfReserved1 = 0;
    bitmapFileHeader.bfReserved2 = 0;
    bitmapFileHeader.bfOffBits = sizeof(BITMAPINFOHEADER)+sizeof(BITMAPFILEHEADER);

    bitmapInfoHeader.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfoHeader.biWidth = X_WIN_SIZE - 0;
    bitmapInfoHeader.biHeight = Y_WIN_SIZE - 0;
    bitmapInfoHeader.biPlanes = 1;
    bitmapInfoHeader.biBitCount = 24;
    bitmapInfoHeader.biCompression = BI_RGB;
    bitmapInfoHeader.biSizeImage = 0;
    bitmapInfoHeader.biXPelsPerMeter = 0; // ?
    bitmapInfoHeader.biYPelsPerMeter = 0; // ?
    bitmapInfoHeader.biClrUsed = 0;
    bitmapInfoHeader.biClrImportant = 0;

    fwrite(&bitmapFileHeader, sizeof(BITMAPFILEHEADER), 1, file);
    fwrite(&bitmapInfoHeader, sizeof(BITMAPINFOHEADER), 1, file);
}
