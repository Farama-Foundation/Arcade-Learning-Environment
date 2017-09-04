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
#include "supported/BankHeist.hpp"
#include "supported/BattleZone.hpp"
#include "supported/BeamRider.hpp"
#include "supported/Berzerk.hpp"
#include "supported/Bowling.hpp"
#include "supported/Boxing.hpp"
#include "supported/Breakout.hpp"
#include "supported/Carnival.hpp"
#include "supported/Centipede.hpp"
#include "supported/ChopperCommand.hpp"
#include "supported/CrazyClimber.hpp"
#include "supported/Defender.hpp"
#include "supported/DemonAttack.hpp"
#include "supported/DonkeyKong.hpp"
#include "supported/DoubleDunk.hpp"
#include "supported/ElevatorAction.hpp"
#include "supported/Enduro.hpp"
#include "supported/FishingDerby.hpp"
#include "supported/Freeway.hpp"
#include "supported/Frogger.hpp"
#include "supported/Frostbite.hpp"
#include "supported/Galaxian.hpp"
#include "supported/Gopher.hpp"
#include "supported/Gravitar.hpp"
#include "supported/Hero.hpp"
#include "supported/IceHockey.hpp"
#include "supported/JamesBond.hpp"
#include "supported/JourneyEscape.hpp"
#include "supported/Kaboom.hpp"
#include "supported/Kangaroo.hpp"
#include "supported/Koolaid.hpp"
#include "supported/KeystoneKapers.hpp"
#include "supported/Kingkong.hpp"
#include "supported/Krull.hpp"
#include "supported/KungFuMaster.hpp"
#include "supported/LaserGates.hpp"
#include "supported/LostLuggage.hpp"
#include "supported/MontezumaRevenge.hpp"
#include "supported/MrDo.hpp"
#include "supported/MsPacman.hpp"
#include "supported/NameThisGame.hpp"
#include "supported/Phoenix.hpp"
#include "supported/Pitfall.hpp"
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
#include "supported/StarGunner.hpp"
#include "supported/Tennis.hpp"
#include "supported/Tetris.hpp"
#include "supported/TimePilot.hpp"
#include "supported/Turmoil.hpp"
#include "supported/Trondead.hpp"
#include "supported/Tutankham.hpp"
#include "supported/UpNDown.hpp"
#include "supported/Venture.hpp"
#include "supported/VideoPinball.hpp"
#include "supported/WizardOfWor.hpp"
#include "supported/YarsRevenge.hpp"
#include "supported/Zaxxon.hpp"


/* list of supported games */
static const RomSettings *roms[]  = {
    new AdventureSettings(),
    new AirRaidSettings(),
    new AlienSettings(),
    new AmidarSettings(),
    new AssaultSettings(),
    new AsterixSettings(),
    new AsteroidsSettings(),
    new AtlantisSettings(),
    new BankHeistSettings(),
    new BattleZoneSettings(),
    new BeamRiderSettings(),
    new BerzerkSettings(),
    new BowlingSettings(),
    new BoxingSettings(),
    new BreakoutSettings(),
    new CarnivalSettings(),
    new CentipedeSettings(),
    new ChopperCommandSettings(),
    new CrazyClimberSettings(),
    new DefenderSettings(),
    new DemonAttackSettings(),
    new DonkeyKongSettings(),
    new DoubleDunkSettings(),
    new ElevatorActionSettings(),
    new EnduroSettings(),
    new FishingDerbySettings(),
    new FreewaySettings(),
    new FroggerSettings(),
    new FrostbiteSettings(),
    new GalaxianSettings(),
    new GopherSettings(),
    new GravitarSettings(),
    new HeroSettings(),
    new IceHockeySettings(),
    new JamesBondSettings(),
    new JourneyEscapeSettings(),
    new KaboomSettings(),
    new KangarooSettings(),
    new KoolaidSettings(),
    new KeystoneKapersSettings(),
    new KingkongSettings(),
    new KrullSettings(),
    new KungFuMasterSettings(),
    new LaserGatesSettings(),
    new LostLuggageSettings(),
    new MontezumaRevengeSettings(),
    new MrDoSettings(),
    new MsPacmanSettings(),
    new NameThisGameSettings(),
    new PhoenixSettings(),
    new PitfallSettings(),
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
    new StarGunnerSettings(),
    new TennisSettings(),
    new TetrisSettings(),
    new TimePilotSettings(),
    new TurmoilSettings(),
    new TrondeadSettings(),
    new TutankhamSettings(),
    new UpNDownSettings(),
    new VentureSettings(),
    new VideoPinballSettings(),
    new WizardOfWorSettings(),
    new YarsRevengeSettings(),
    new ZaxxonSettings(),
};


/* looks for the RL wrapper corresponding to a particular rom title */
RomSettings *buildRomRLWrapper(const std::string &rom) {

    size_t slash_ind = rom.find_last_of("/\\");
    std::string rom_str = rom.substr(slash_ind + 1);
    size_t dot_idx = rom_str.find_first_of(".");
    rom_str = rom_str.substr(0, dot_idx);
    std::transform(rom_str.begin(), rom_str.end(), rom_str.begin(), ::tolower);

    for (size_t i=0; i < sizeof(roms)/sizeof(roms[0]); i++) {
        if (rom_str == roms[i]->rom()) return roms[i]->clone();
    }

    return NULL;
}

