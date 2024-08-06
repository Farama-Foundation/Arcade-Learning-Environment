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

#include <filesystem>

#include "ale/games/Roms.hpp"
#include "ale/games/RomUtils.hpp"

// include the game implementations
#include "ale/games/supported/Adventure.hpp"
#include "ale/games/supported/AirRaid.hpp"
#include "ale/games/supported/Alien.hpp"
#include "ale/games/supported/Amidar.hpp"
#include "ale/games/supported/Assault.hpp"
#include "ale/games/supported/Asterix.hpp"
#include "ale/games/supported/Asteroids.hpp"
#include "ale/games/supported/Atlantis.hpp"
#include "ale/games/supported/Atlantis2.hpp"
#include "ale/games/supported/Backgammon.hpp"
#include "ale/games/supported/BankHeist.hpp"
#include "ale/games/supported/BasicMath.hpp"
#include "ale/games/supported/BattleZone.hpp"
#include "ale/games/supported/BeamRider.hpp"
#include "ale/games/supported/Berzerk.hpp"
#include "ale/games/supported/Blackjack.hpp"
#include "ale/games/supported/Bowling.hpp"
#include "ale/games/supported/Boxing.hpp"
#include "ale/games/supported/Breakout.hpp"
#include "ale/games/supported/Carnival.hpp"
#include "ale/games/supported/Casino.hpp"
#include "ale/games/supported/Centipede.hpp"
#include "ale/games/supported/ChopperCommand.hpp"
#include "ale/games/supported/CrazyClimber.hpp"
#include "ale/games/supported/Crossbow.hpp"
#include "ale/games/supported/DarkChambers.hpp"
#include "ale/games/supported/Defender.hpp"
#include "ale/games/supported/DemonAttack.hpp"
#include "ale/games/supported/DonkeyKong.hpp"
#include "ale/games/supported/DoubleDunk.hpp"
#include "ale/games/supported/Earthworld.hpp"
#include "ale/games/supported/ElevatorAction.hpp"
#include "ale/games/supported/Enduro.hpp"
#include "ale/games/supported/Entombed.hpp"
#include "ale/games/supported/Et.hpp"
#include "ale/games/supported/FishingDerby.hpp"
#include "ale/games/supported/FlagCapture.hpp"
#include "ale/games/supported/Freeway.hpp"
#include "ale/games/supported/Frogger.hpp"
#include "ale/games/supported/Frostbite.hpp"
#include "ale/games/supported/Galaxian.hpp"
#include "ale/games/supported/Gopher.hpp"
#include "ale/games/supported/Gravitar.hpp"
#include "ale/games/supported/Hangman.hpp"
#include "ale/games/supported/HauntedHouse.hpp"
#include "ale/games/supported/Hero.hpp"
#include "ale/games/supported/HumanCannonball.hpp"
#include "ale/games/supported/IceHockey.hpp"
#include "ale/games/supported/JamesBond.hpp"
#include "ale/games/supported/JourneyEscape.hpp"
#include "ale/games/supported/Kaboom.hpp"
#include "ale/games/supported/Kangaroo.hpp"
#include "ale/games/supported/KeystoneKapers.hpp"
#include "ale/games/supported/Kingkong.hpp"
#include "ale/games/supported/Klax.hpp"
#include "ale/games/supported/Koolaid.hpp"
#include "ale/games/supported/Krull.hpp"
#include "ale/games/supported/KungFuMaster.hpp"
#include "ale/games/supported/LaserGates.hpp"
#include "ale/games/supported/LostLuggage.hpp"
#include "ale/games/supported/MarioBros.hpp"
#include "ale/games/supported/MiniatureGolf.hpp"
#include "ale/games/supported/MontezumaRevenge.hpp"
#include "ale/games/supported/MrDo.hpp"
#include "ale/games/supported/MsPacman.hpp"
#include "ale/games/supported/NameThisGame.hpp"
#include "ale/games/supported/Othello.hpp"
#include "ale/games/supported/Pacman.hpp"
#include "ale/games/supported/Phoenix.hpp"
#include "ale/games/supported/Pitfall.hpp"
#include "ale/games/supported/Pitfall2.hpp"
#include "ale/games/supported/Pong.hpp"
#include "ale/games/supported/Pooyan.hpp"
#include "ale/games/supported/PrivateEye.hpp"
#include "ale/games/supported/QBert.hpp"
#include "ale/games/supported/RiverRaid.hpp"
#include "ale/games/supported/RoadRunner.hpp"
#include "ale/games/supported/RoboTank.hpp"
#include "ale/games/supported/Seaquest.hpp"
#include "ale/games/supported/SirLancelot.hpp"
#include "ale/games/supported/Skiing.hpp"
#include "ale/games/supported/Solaris.hpp"
#include "ale/games/supported/SpaceInvaders.hpp"
#include "ale/games/supported/SpaceWar.hpp"
#include "ale/games/supported/StarGunner.hpp"
#include "ale/games/supported/Superman.hpp"
#include "ale/games/supported/Surround.hpp"
#include "ale/games/supported/Tennis.hpp"
#include "ale/games/supported/Tetris.hpp"
#include "ale/games/supported/TicTacToe3d.hpp"
#include "ale/games/supported/TimePilot.hpp"
#include "ale/games/supported/Trondead.hpp"
#include "ale/games/supported/Turmoil.hpp"
#include "ale/games/supported/Tutankham.hpp"
#include "ale/games/supported/UpNDown.hpp"
#include "ale/games/supported/Venture.hpp"
#include "ale/games/supported/VideoCheckers.hpp"
#include "ale/games/supported/VideoChess.hpp"
#include "ale/games/supported/VideoCube.hpp"
#include "ale/games/supported/VideoPinball.hpp"
#include "ale/games/supported/WizardOfWor.hpp"
#include "ale/games/supported/WordZapper.hpp"
#include "ale/games/supported/YarsRevenge.hpp"
#include "ale/games/supported/Zaxxon.hpp"

namespace fs = std::filesystem;

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
    new WizardOfWorSettings(),
    new WordZapperSettings(),
    new YarsRevengeSettings(),
    new ZaxxonSettings(),
};
/* looks for the RL wrapper corresponding to a particular rom filename,
 * and optionally md5. returns null if neither match */
RomSettings* buildRomRLWrapper(const fs::path& rom, const std::string rom_md5 = std::string()){
  // Stem is filename excluding the extension.
  std::string rom_str = rom.stem().string();
  std::transform(rom_str.begin(), rom_str.end(), rom_str.begin(), ::tolower);

  for (size_t i = 0; i < sizeof(roms) / sizeof(roms[0]); i++) {
    if (rom_md5 == roms[i]->md5() || rom_str == roms[i]->rom())
      return roms[i]->clone();
  }
  return NULL;
}

}  // namespace ale
