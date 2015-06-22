/* 
* Copyright (C) 2008, Brian Tanner

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

/** This example was *minimally* adapted from the SkeletonAgent code from 
  * Brian Tanner. The chief modification is the two random actions, instead
  * of one. The following is required to execute this code:
  *
  *  - RL-Glue core
  *  - RL-Glue C/C++ codec
  */

#include <stdio.h>	/* for printf */
#include <rlglue/RL_glue.h> /* RL_ function prototypes and RL-Glue types */
	
int whichEpisode=0;

// This uses RL-Glue to run a single episode.
void runEpisode(int stepLimit) {        
    int terminal=RL_episode(stepLimit);
	printf("Episode %d\t %d steps \t%f total reward\t %d natural end \n",whichEpisode,RL_num_steps(),RL_return(), terminal);
	whichEpisode++;
}

int main(int argc, char *argv[]) {
	const char* task_spec;
	const reward_observation_action_terminal_t *stepResponse;
	const observation_action_t *startResponse;

	printf("\n\nExperiment starting up!\n");


	task_spec=RL_init();
	printf("RL_init called, the environment sent task spec: %s\n",task_spec);

	// RL_env_message and RL_agent_message may be used to communicate with the environment
    // and agent, respectively. See RL-Glue documentation for details.
	// const char* responseMessage;
	// responseMessage=RL_agent_message("what is your name?");

	printf("\n\n----------Running a few episodes----------\n");
	// Use the RL-Glue-provided RL_episode to run a few episodes of ALE. 
    // 0 means no limit at all.
	runEpisode(10000);
	runEpisode(0);
	runEpisode(0);
	runEpisode(0);
	runEpisode(0);
	RL_cleanup();

	printf("\n\n----------Stepping through an episode----------\n");
	// The following demonstrates how to step through an episode. 
    task_spec=RL_init();

	// Start the episode
	startResponse=RL_start();
	printf("First action was: %d\n", startResponse->action->intArray[0]);

    // Run one step	
	stepResponse=RL_step();
	
	// Run until end of episode
	while(stepResponse->terminal != 1) {
		stepResponse=RL_step();
	}

    // Demonstrates other RL-Glue functionality.
	printf("It ran for %d steps, total reward was: %f\n",RL_num_steps(), RL_return());
	RL_cleanup();


	return 0;
}
