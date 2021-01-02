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

#include "games/Roms.hpp"
#include "games/RomUtils.hpp"

// include the game implementations
#include "games/supported/Adventure.hpp"
#include "games/supported/AirRaid.hpp"
#include "games/supported/Alien.hpp"
#include "games/supported/Amidar.hpp"
#include "games/supported/Assault.hpp"
#include "games/supported/Asterix.hpp"
#include "games/supported/Asteroids.hpp"
#include "games/supported/Atlantis.hpp"
#include "games/supported/Atlantis2.hpp"
#include "games/supported/Backgammon.hpp"
#include "games/supported/BankHeist.hpp"
#include "games/supported/BasicMath.hpp"
#include "games/supported/BattleZone.hpp"
#include "games/supported/BeamRider.hpp"
#include "games/supported/Berzerk.hpp"
#include "games/supported/Blackjack.hpp"
#include "games/supported/Bowling.hpp"
#include "games/supported/Boxing.hpp"
#include "games/supported/Breakout.hpp"
#include "games/supported/Carnival.hpp"
#include "games/supported/Casino.hpp"
#include "games/supported/Centipede.hpp"
#include "games/supported/ChopperCommand.hpp"
#include "games/supported/CrazyClimber.hpp"
#include "games/supported/Crossbow.hpp"
#include "games/supported/DarkChambers.hpp"
#include "games/supported/Defender.hpp"
#include "games/supported/DemonAttack.hpp"
#include "games/supported/DonkeyKong.hpp"
#include "games/supported/DoubleDunk.hpp"
#include "games/supported/Earthworld.hpp"
#include "games/supported/ElevatorAction.hpp"
#include "games/supported/Enduro.hpp"
#include "games/supported/Entombed.hpp"
#include "games/supported/Et.hpp"
#include "games/supported/FishingDerby.hpp"
#include "games/supported/FlagCapture.hpp"
#include "games/supported/Freeway.hpp"
#include "games/supported/Frogger.hpp"
#include "games/supported/Frostbite.hpp"
#include "games/supported/Galaxian.hpp"
#include "games/supported/Gopher.hpp"
#include "games/supported/Gravitar.hpp"
#include "games/supported/Hangman.hpp"
#include "games/supported/HauntedHouse.hpp"
#include "games/supported/Hero.hpp"
#include "games/supported/HumanCannonball.hpp"
#include "games/supported/IceHockey.hpp"
#include "games/supported/JamesBond.hpp"
#include "games/supported/JourneyEscape.hpp"
#include "games/supported/Kaboom.hpp"
#include "games/supported/Kangaroo.hpp"
#include "games/supported/KeystoneKapers.hpp"
#include "games/supported/Kingkong.hpp"
#include "games/supported/Klax.hpp"
#include "games/supported/Koolaid.hpp"
#include "games/supported/Krull.hpp"
#include "games/supported/KungFuMaster.hpp"
#include "games/supported/LaserGates.hpp"
#include "games/supported/LostLuggage.hpp"
#include "games/supported/MarioBros.hpp"
#include "games/supported/MiniatureGolf.hpp"
#include "games/supported/MontezumaRevenge.hpp"
#include "games/supported/MrDo.hpp"
#include "games/supported/MsPacman.hpp"
#include "games/supported/NameThisGame.hpp"
#include "games/supported/Othello.hpp"
#include "games/supported/Pacman.hpp"
#include "games/supported/Phoenix.hpp"
#include "games/supported/Pitfall.hpp"
#include "games/supported/Pitfall2.hpp"
#include "games/supported/Pong.hpp"
#include "games/supported/Pooyan.hpp"
#include "games/supported/PrivateEye.hpp"
#include "games/supported/QBert.hpp"
#include "games/supported/RiverRaid.hpp"
#include "games/supported/RoadRunner.hpp"
#include "games/supported/RoboTank.hpp"
#include "games/supported/Seaquest.hpp"
#include "games/supported/SirLancelot.hpp"
#include "games/supported/Skiing.hpp"
#include "games/supported/Solaris.hpp"
#include "games/supported/SpaceInvaders.hpp"
#include "games/supported/SpaceWar.hpp"
#include "games/supported/StarGunner.hpp"
#include "games/supported/Superman.hpp"
#include "games/supported/Surround.hpp"
#include "games/supported/Tennis.hpp"
#include "games/supported/Tetris.hpp"
#include "games/supported/TicTacToe3d.hpp"
#include "games/supported/TimePilot.hpp"
#include "games/supported/Trondead.hpp"
#include "games/supported/Turmoil.hpp"
#include "games/supported/Tutankham.hpp"
#include "games/supported/UpNDown.hpp"
#include "games/supported/Venture.hpp"
#include "games/supported/VideoCheckers.hpp"
#include "games/supported/VideoChess.hpp"
#include "games/supported/VideoCube.hpp"
#include "games/supported/VideoPinball.hpp"
#include "games/supported/WizardOfWor.hpp"
#include "games/supported/WordZapper.hpp"
#include "games/supported/YarsRevenge.hpp"
#include "games/supported/Zaxxon.hpp"

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
