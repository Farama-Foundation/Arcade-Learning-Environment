#include "ale_interface.hpp"
#include <fstream>
#include <vector>

using namespace ale;
using namespace std;

std::vector<std::string> two_player_games;
void init_two_player_fnames(){
  std::vector<std::string> two_player_fnames = {
    "backgammon",
    "blackjack",
    "boxing",
    "casino",
    "double_dunk",
    "entombed",
    "fishing_derby",
    "flag_capture",
    "ice_hockey",
    "lost_luggage",
    "othello",
    "pong",
    "space_invaders",
    "space_war",
    "surround",
    "tennis",
    "video_checkers",
    "wizard_of_wor",
  };
  std::string main_path = "/home/benblack/anaconda3/lib/python3.7/site-packages/ale_py/ROM/";
  for(std::string fname : two_player_fnames){
    std::string new_fname = main_path + fname + "/" + fname + ".bin";
    two_player_games.push_back(new_fname);
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
  return interface.supportsTwoPlayers();
}
std::size_t hash_vec(std::vector<uint8_t> const& vec) {
  std::size_t seed = vec.size();
  for(uint8_t i : vec) {
    seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}
size_t play_sequence(ALEInterface & interface,int sequence_num){
  size_t hashCode = 0;
  std::vector<unsigned char> output_rgb_buffer;
  ActionVect min_actionsp1 = interface.getMinimalActionSet();
  Action p1_action = min_actionsp1[0];
  ActionVect min_actionsp2 = interface.getMinimalActionSetP2();
  for(int j = 0; j < 2000; j++){

    int action_p1 = (j/64)%min_actionsp1.size();
    int action_p2 = sequence_num == 0 ? (j/4)%min_actionsp2.size() : (j/64)%min_actionsp2.size();

    interface.act2P(min_actionsp1[action_p1], min_actionsp2[action_p2]);

    interface.getScreenRGB(output_rgb_buffer);
    hashCode = hash_vec(output_rgb_buffer);
  }
  return hashCode;
}
bool test_two_player_controlability(std::string fname){
  int seed = 123982;
  size_t hashs[2] = {0,0};
  for(int i = 0; i < 2; i++){
    ALEInterface interface;
    interface.setInt("random_seed", seed);
    interface.loadROM(fname);
    ModeVect modes = interface.get2PlayerModes();
    interface.setMode(modes[0]);
    hashs[i] = play_sequence(interface,i);
  }
  return hashs[0] != hashs[1];
}
int main(){
  init_two_player_fnames();
  for(std::string fname : two_player_games){
    size_t last_slash = fname.find_last_of('/');
    std::string binname(fname.begin() + last_slash, fname.end());
    if(!test_env_exists(fname)){
        cout << "failed to load file!\n";
        cout << fname << "\n";
        exit(-1);
    }
    else if(!test_two_player(fname)){
      cout << binname << " environment not two player\n";
    }
    else if(!test_two_player_controlability(fname)){
      cout << binname << " environment not controllable\n";
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
