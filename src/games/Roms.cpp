/* *****************************************************************************
 * A.L.E (Arcade Learning Environment)
 * Copyright (c) 2009-2013 by Yavar Naddaf, Joel Veness, Marc G. Bellemare and
 *   the Reinforcement Learning and Artificial Intelligence Laboratory
 * Released under the GNU General Public License; see License.txt for details.
 *
 * Based on: Stella  --  "An Atari 2600 VCS Emulator"
 * Copyright (c) 1995-2007 by Bradford W. Mott and the Stella team
 *
 * *****************************************************************************
 */

#include <fstream>

#include "Roms.hpp"
#include "RomUtils.hpp"

// include the game implementations
#include "supported/Adventure.hpp"
#include "supported/AirRaid.hpp"
#include "supported/Alien.hpp"
#include "supported/Amidar.hpp"
#include "supported/Assault.hpp"
#include "supported/Asterix.hpp"
#include "supported/Asteroids.hpp"
#include "supported/Atlantis.hpp"
#include "supported/Atlantis2.hpp"
#include "supported/Backgammon.hpp"
#include "supported/BankHeist.hpp"
#include "supported/BasicMath.hpp"
#include "supported/BattleZone.hpp"
#include "supported/BeamRider.hpp"
#include "supported/Berzerk.hpp"
#include "supported/Blackjack.hpp"
#include "supported/Bowling.hpp"
#include "supported/Boxing.hpp"
#include "supported/Breakout.hpp"
#include "supported/Carnival.hpp"
#include "supported/Casino.hpp"
#include "supported/Centipede.hpp"
#include "supported/ChopperCommand.hpp"
#include "supported/Combat.hpp"
#include "supported/CrazyClimber.hpp"
#include "supported/Crossbow.hpp"
#include "supported/DarkChambers.hpp"
#include "supported/Defender.hpp"
#include "supported/DemonAttack.hpp"
#include "supported/DonkeyKong.hpp"
#include "supported/DoubleDunk.hpp"
#include "supported/Earthworld.hpp"
#include "supported/ElevatorAction.hpp"
#include "supported/Enduro.hpp"
#include "supported/Entombed.hpp"
#include "supported/Et.hpp"
#include "supported/FishingDerby.hpp"
#include "supported/FlagCapture.hpp"
#include "supported/Freeway.hpp"
#include "supported/Frogger.hpp"
#include "supported/Frostbite.hpp"
#include "supported/Galaxian.hpp"
#include "supported/Gopher.hpp"
#include "supported/Gravitar.hpp"
#include "supported/Hangman.hpp"
#include "supported/HauntedHouse.hpp"
#include "supported/Hero.hpp"
#include "supported/HumanCannonball.hpp"
#include "supported/IceHockey.hpp"
#include "supported/JamesBond.hpp"
#include "supported/JourneyEscape.hpp"
#include "supported/Joust.hpp"
#include "supported/Kaboom.hpp"
#include "supported/Kangaroo.hpp"
#include "supported/KeystoneKapers.hpp"
#include "supported/Kingkong.hpp"
#include "supported/Klax.hpp"
#include "supported/Koolaid.hpp"
#include "supported/Krull.hpp"
#include "supported/KungFuMaster.hpp"
#include "supported/LaserGates.hpp"
#include "supported/LostLuggage.hpp"
#include "supported/MarioBros.hpp"
#include "supported/MazeCraze.hpp"
#include "supported/MiniatureGolf.hpp"
#include "supported/MontezumaRevenge.hpp"
#include "supported/MrDo.hpp"
#include "supported/MsPacman.hpp"
#include "supported/NameThisGame.hpp"
#include "supported/Othello.hpp"
#include "supported/Pacman.hpp"
#include "supported/Phoenix.hpp"
#include "supported/Pitfall.hpp"
#include "supported/Pitfall2.hpp"
#include "supported/Pong.hpp"
#include "supported/Pooyan.hpp"
#include "supported/PrivateEye.hpp"
#include "supported/QBert.hpp"
#include "supported/RiverRaid.hpp"
#include "supported/RoadRunner.hpp"
#include "supported/RoboTank.hpp"
#include "supported/Seaquest.hpp"
#include "supported/SirLancelot.hpp"
#include "supported/Skiing.hpp"
#include "supported/Solaris.hpp"
#include "supported/SpaceInvaders.hpp"
#include "supported/SpaceWar.hpp"
#include "supported/StarGunner.hpp"
#include "supported/Superman.hpp"
#include "supported/Surround.hpp"
#include "supported/Tennis.hpp"
#include "supported/Tetris.hpp"
#include "supported/TicTacToe3d.hpp"
#include "supported/TimePilot.hpp"
#include "supported/Trondead.hpp"
#include "supported/Turmoil.hpp"
#include "supported/Tutankham.hpp"
#include "supported/UpNDown.hpp"
#include "supported/Venture.hpp"
#include "supported/VideoCheckers.hpp"
#include "supported/VideoChess.hpp"
#include "supported/VideoCube.hpp"
#include "supported/VideoPinball.hpp"
#include "supported/Warlords.hpp"
#include "supported/WizardOfWor.hpp"
#include "supported/WordZapper.hpp"
#include "supported/YarsRevenge.hpp"
#include "supported/Zaxxon.hpp"

#include "emucore/MD5.hxx"

namespace ale {

/* list of supported games */
static const RomSettings* roms[] = {
    new AdventureSettings(),
    new AirRaidSettings(),
    new AlienSettings(),
    new AmidarSettings(),
    new AssaultSettings(),
    new AsterixSettings(),
    new AsteroidsSettings(),
    new AtlantisSettings(),
    new Atlantis2Settings(),
    new BackgammonSettings(),
    new BankHeistSettings(),
    new BasicMathSettings(),
    new BattleZoneSettings(),
    new BeamRiderSettings(),
    new BerzerkSettings(),
    new BlackjackSettings(),
    new BowlingSettings(),
    new BoxingSettings(),
    new BreakoutSettings(),
    new CarnivalSettings(),
    new CasinoSettings(),
    new CentipedeSettings(),
    new ChopperCommandSettings(),
    new CombatSettings(),
    new CrazyClimberSettings(),
    new CrossbowSettings(),
    new DarkChambersSettings(),
    new DefenderSettings(),
    new DemonAttackSettings(),
    new DonkeyKongSettings(),
    new DoubleDunkSettings(),
    new EarthworldSettings(),
    new ElevatorActionSettings(),
    new EnduroSettings(),
    new EntombedSettings(),
    new EtSettings(),
    new FishingDerbySettings(),
    new FlagCaptureSettings(),
    new FreewaySettings(),
    new FroggerSettings(),
    new FrostbiteSettings(),
    new GalaxianSettings(),
    new GopherSettings(),
    new GravitarSettings(),
    new HangmanSettings(),
    new HauntedHouseSettings(),
    new HeroSettings(),
    new HumanCannonballSettings(),
    new IceHockeySettings(),
    new JamesBondSettings(),
    new JourneyEscapeSettings(),
    new JoustSettings(),
    new KaboomSettings(),
    new KangarooSettings(),
    new KoolaidSettings(),
    new KeystoneKapersSettings(),
    new KingkongSettings(),
    new KlaxSettings(),
    new KrullSettings(),
    new KungFuMasterSettings(),
    new LaserGatesSettings(),
    new LostLuggageSettings(),
    new MarioBrosSettings(),
    new MazeCrazeSettings(),
    new MiniatureGolfSettings(),
    new MontezumaRevengeSettings(),
    new MrDoSettings(),
    new MsPacmanSettings(),
    new NameThisGameSettings(),
    new OthelloSettings(),
    new PacmanSettings(),
    new PhoenixSettings(),
    new PitfallSettings(),
    new Pitfall2Settings(),
    new PongSettings(),
    new PooyanSettings(),
    new PrivateEyeSettings(),
    new QBertSettings(),
    new RiverRaidSettings(),
    new RoadRunnerSettings(),
    new RoboTankSettings(),
    new SeaquestSettings(),
    new SirLancelotSettings(),
    new SkiingSettings(),
    new SolarisSettings(),
    new SpaceInvadersSettings(),
    new SpaceWarSettings(),
    new StarGunnerSettings(),
    new SupermanSettings(),
    new SurroundSettings(),
    new TennisSettings(),
    new TetrisSettings(),
    new TicTacToe3dSettings(),
    new TimePilotSettings(),
    new TurmoilSettings(),
    new TrondeadSettings(),
    new TutankhamSettings(),
    new UpNDownSettings(),
    new VentureSettings(),
    new VideoCheckersSettings(),
    new VideoChessSettings(),
    new VideoCubeSettings(),
    new VideoPinballSettings(),
    new WarlordsSettings(),
    new WizardOfWorSettings(),
    new WordZapperSettings(),
    new YarsRevengeSettings(),
    new ZaxxonSettings(),
};

/* looks for the RL wrapper corresponding to a particular rom title */
RomSettings* buildRomRLWrapper(const std::string& rom) {
  size_t slash_ind = rom.find_last_of("/\\");
  std::string rom_str = rom.substr(slash_ind + 1);
  size_t dot_idx = rom_str.find_first_of(".");
  rom_str = rom_str.substr(0, dot_idx);
  std::transform(rom_str.begin(), rom_str.end(), rom_str.begin(), ::tolower);

  std::ifstream romfile(rom);
  std::string str((std::istreambuf_iterator<char>(romfile)),
                   std::istreambuf_iterator<char>());
  std::string md5_val = MD5((const unsigned char*)str.data(),str.size());
  for (size_t i = 0; i < sizeof(roms) / sizeof(roms[0]); i++) {
    if (md5_val == roms[i]->md5())
      return roms[i]->clone();
  }
  for (size_t i = 0; i < sizeof(roms) / sizeof(roms[0]); i++) {
    if (rom_str == roms[i]->rom())
      return roms[i]->clone();
  }

  return NULL;
}

}  // namespace ale
