MODULE := src/games

MODULE_OBJS := \
	src/games/Roms.o \
	src/games/RomSettings.o \
	src/games/RomUtils.o \
	src/games/supported/Adventure.o \
	src/games/supported/AirRaid.o \
	src/games/supported/Alien.o \
	src/games/supported/Amidar.o \
	src/games/supported/Assault.o \
	src/games/supported/Asterix.o \
	src/games/supported/Asteroids.o \
	src/games/supported/Atlantis.o \
	src/games/supported/BankHeist.o \
	src/games/supported/BattleZone.o \
	src/games/supported/BeamRider.o \
	src/games/supported/Berzerk.o \
	src/games/supported/Bowling.o \
	src/games/supported/Boxing.o \
	src/games/supported/Breakout.o \
	src/games/supported/Carnival.o \
	src/games/supported/Centipede.o \
	src/games/supported/ChopperCommand.o \
	src/games/supported/CrazyClimber.o \
	src/games/supported/Defender.o \
	src/games/supported/DemonAttack.o \
	src/games/supported/DonkeyKong.o \
	src/games/supported/DoubleDunk.o \
	src/games/supported/ElevatorAction.o \
	src/games/supported/Enduro.o \
	src/games/supported/FishingDerby.o \
	src/games/supported/Freeway.o \
	src/games/supported/Frogger.o \
	src/games/supported/Frostbite.o \
	src/games/supported/Galaxian.o \
	src/games/supported/Gopher.o \
	src/games/supported/Gravitar.o \
	src/games/supported/Hero.o \
	src/games/supported/IceHockey.o \
	src/games/supported/JamesBond.o \
	src/games/supported/JourneyEscape.o \
	src/games/supported/Kaboom.o \
	src/games/supported/Kangaroo.o \
	src/games/supported/Koolaid.o \
	src/games/supported/KeystoneKapers.o \
	src/games/supported/Kingkong.o \
	src/games/supported/Krull.o \
	src/games/supported/KungFuMaster.o \
	src/games/supported/LaserGates.o \
	src/games/supported/LostLuggage.o \
	src/games/supported/MontezumaRevenge.o \
	src/games/supported/MrDo.o \
	src/games/supported/MsPacman.o \
	src/games/supported/NameThisGame.o \
	src/games/supported/Phoenix.o \
	src/games/supported/Pitfall.o \
	src/games/supported/Pong.o \
	src/games/supported/Pooyan.o \
	src/games/supported/PrivateEye.o \
	src/games/supported/QBert.o \
	src/games/supported/RiverRaid.o \
	src/games/supported/RoadRunner.o \
	src/games/supported/RoboTank.o \
	src/games/supported/Seaquest.o \
	src/games/supported/SirLancelot.o \
	src/games/supported/Skiing.o \
	src/games/supported/Solaris.o \
	src/games/supported/SpaceInvaders.o \
	src/games/supported/StarGunner.o \
	src/games/supported/Tennis.o \
	src/games/supported/Tetris.o \
	src/games/supported/TimePilot.o \
	src/games/supported/Turmoil.o \
	src/games/supported/Tutankham.o \
	src/games/supported/Trondead.o \
	src/games/supported/UpNDown.o \
	src/games/supported/Venture.o \
	src/games/supported/VideoPinball.o \
	src/games/supported/WizardOfWor.o \
	src/games/supported/YarsRevenge.o \
	src/games/supported/Zaxxon.o \

MODULE_DIRS += \
	src/games

# Include common rules 
include $(srcdir)/common.rules
