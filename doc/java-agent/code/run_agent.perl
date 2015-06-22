#!/usr/bin/perl
use IPC::Open2;

# -----------------------------------
#          test configuration
# -----------------------------------

# Original script by Joel Veness, modified by Marc G. Bellemare. Runs a 
#  Java agent for a given number of episodes. If the requested number of 
#  episodes is 0, the script runs forever. The default number of episodes 
#  is 1.

$agentJarFile="dist/ALEJavaAgent.jar";

# disables buffered IO
$| = 1;

$OS = $^O;
$numArgs = $#ARGV + 1;
die "Usage: $0 <rom> [-export_frames]" if ($numArgs < 1);

$debug_mode = 1; # 0 off, 1 some information, 2 verbose
$rom = $ARGV[0];
if ($numArgs > 1) {
  $arg=$ARGV[1]; 
}
$num_episodes = 1;

$env_cmd = "./ale -game_controller fifo roms/$rom.bin";
$agent_cmd = "java -Xmx1024M -jar $agentJarFile " . $arg; 

if ($OS eq "linux" or $OS eq "darwin") {
  # platform specific code
}


# -----------------------------------
#            main loop
# -----------------------------------
    
		local (*AGENT_READ, *AGENT_WRITE, *AGENT_ERR);
    local (*ENV_READ, *ENV_WRITE, *ENV_ERR);
    
		$pid_env    = open2(\*ENV_READ, \*ENV_WRITE, $env_cmd);
    $pid_agent  = open2(\*AGENT_READ, \*AGENT_WRITE, $agent_cmd);

    print "Started Agent with PID: $pid_agent\n" if $debug_mode > 1;
    print "Started Environment with PID:   $pid_env\n" if $debug_mode > 1;
    


$episode = 1;

$total_reward = 0;
$step = 1;
$episode_on = 0;
$ep_start_time = time;

ALL_EPISODES: {
    do {
        # read from environment
        die "environment terminated unexpectedly" unless kill(0, $pid_env);
	$l = <ENV_READ>;
	
	# send to agent
	print AGENT_WRITE $l;
        
	# extract reward and terminal status, skip during handshaking
	if ($episode_on) {
	    @f = split /:/, $l; 
  	    $tok = $f[$#f-1];
	    @g = split /,/, $tok;
 	    $terminate = $g[0];	    
	    $total_reward += $g[1];

	    if ($terminate == 1) {
	    	$ep_end_time = time;
				$episode_on = 0;
				$total_time = time - $ep_start_time;
				print "Episode $episode $total_reward $total_time $step\n";

				$ep_start_time = time;
				$total_reward = 0;
				$step = 1;

				last ALL_EPISODES if ++$episode > $num_episodes && $num_episodes > 0;
	    }
	    print "Time: $c, Reward: $g[1],\n" if $debug_mode >= 2;
	}

  # read from agent, send to environment
  die "agent terminated unexpectedly" unless kill(0, $pid_agent);
  $a = <AGENT_READ>; 
  print ENV_WRITE $a;
  print "Action: $a\n" if $debug_mode >= 2;

 	# On system reset start the episode
	$player_a_act = (split /,/, $a)[0];
	$episode_on = 1 if $player_a_act == 45;

  die "agent failed to send back an action" if length($player_a_act) == 0;
	$step++;
  } while (1);
}

# Somewhat ugly; we have to write to the stream before we ask the agent
#  for an action because of blocking sockets
print AGENT_WRITE "DIE\n";
$a = <AGENT_READ>;

# terminate agent/environment
waitpid($pid_agent, 0);
close ENV_READ; close ENV_WRITE;
kill $pid_env, 9; 
