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

/* Run One Episode of length maximum cutOff*/
void runEpisode(int stepLimit) {        
    int terminal=RL_episode(stepLimit);
	printf("Episode %d\t %d steps \t%f total reward\t %d natural end \n",whichEpisode,RL_num_steps(),RL_return(), terminal);
	whichEpisode++;
}

int main(int argc, char *argv[]) {
	const char* task_spec;
	const char* responseMessage;
	const reward_observation_action_terminal_t *stepResponse;
	const observation_action_t *startResponse;

	printf("\n\nExperiment starting up!\n");


	task_spec=RL_init();
	printf("RL_init called, the environment sent task spec: %s\n",task_spec);

	printf("\n\n----------Sending some sample messages----------\n");
	/*Talk to the agent and environment a bit...*/
	responseMessage=RL_agent_message("what is your name?");
	printf("Agent responded to \"what is your name?\" with: %s\n",responseMessage);
	responseMessage=RL_agent_message("If at first you don't succeed; call it version 1.0");
	printf("Agent responded to \"If at first you don't succeed; call it version 1.0\" with: %s\n\n",responseMessage);

	responseMessage=RL_env_message("what is your name?");
	printf("Environment responded to \"what is your name?\" with: %s\n",responseMessage);
	responseMessage=RL_env_message("If at first you don't succeed; call it version 1.0");
	printf("Environment responded to \"If at first you don't succeed; call it version 1.0\" with: %s\n",responseMessage);

	printf("\n\n----------Running a few episodes----------\n");
	/* Remember that stepLimit of 0 means there is no limit at all!*/
	runEpisode(10000);
	runEpisode(0);
	runEpisode(0);
	runEpisode(0);
	runEpisode(0);
	RL_cleanup();

	printf("\n\n----------Stepping through an episode----------\n");
	/*We could also start over and do another experiment */
	task_spec=RL_init();

	/*We could run one step at a time instead of one episode at a time */
	/*Start the episode */
	startResponse=RL_start();
	printf("First observation and action were: %d %d\n",startResponse->observation->intArray[0],startResponse->action->intArray[0]);

	/*Run one step */
	stepResponse=RL_step();
	
	/*Run until the episode ends*/
	while(stepResponse->terminal!=1){
		stepResponse=RL_step();
		if(stepResponse->terminal!=1){
			/*Could optionally print state,action pairs */
			/*printf("(%d,%d) ",stepResponse.o.intArray[0],stepResponse.a.intArray[0]);*/
		}
	}
	
	printf("\n\n----------Summary----------\n");
	

	printf("It ran for %d steps, total reward was: %f\n",RL_num_steps(),RL_return());
	RL_cleanup();


	return 0;
}
