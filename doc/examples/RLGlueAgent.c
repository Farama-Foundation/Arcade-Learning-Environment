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

#include <stdio.h>  /* for printf */
#include <string.h> /* for strcmp */
#include <time.h> /*for time()*/
#include <rlglue/Agent_common.h> /* agent_ function prototypes and RL-Glue types */
#include <rlglue/utils/C/RLStruct_util.h> /* helpful functions for allocating structs and cleaning them up */


action_t this_action;
action_t last_action;

observation_t *last_observation=0;

int randInRange(int max){
	double r, x;
	r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
   	x = (r * (max+1));
	return (int)x;
}

void agent_init(const char* task_spec)
{
	/*Seed the random number generator*/

	srand(time(0));
	/*Here is where you might allocate storage for parameters (value function or policy, last action, last observation, etc)*/
	
	/*Here you would parse the task spec if you felt like it*/
	
	/*Allocate memory for a one-dimensional integer action using utility functions from RLStruct_util*/
	allocateRLStruct(&this_action,2,0,0);
	last_observation=allocateRLStructPointer(0,0,0);
	
	/* That is equivalent to:
			 this_action.numInts     =  1;
			 this_action.intArray    = (int*)calloc(1,sizeof(int));
			 this_action.numDoubles  = 0;
			 this_action.doubleArray = 0;
			 this_action.numChars    = 0;
			 this_action.charArray   = 0;
	*/
}

const action_t *agent_start(const observation_t *this_observation) {
	/* This agent always returns a random number, either 0 or 1 for its action */
	int theIntAction=randInRange(1);
	this_action.intArray[0]=theIntAction;

	/* In a real action you might want to store the last observation and last action*/
	replaceRLStruct(&this_action, &last_action);
	replaceRLStruct(this_observation, last_observation);
	
	return &this_action;
}

const action_t *agent_step(double reward, const observation_t *this_observation) {
  int row, col;

	/* This agent  returns 0 or 1 randomly for its action */
	this_action.intArray[0] = randInRange(17);
  this_action.intArray[1] = randInRange(17) + 18;

  /* Print out the RAM */
  for (row = 0; row < 8; row++) {
    for (col = 0; col < 16; col++)
      fprintf (stderr, "%2x ", this_observation->intArray[col + row*16]);
    fprintf (stderr, "\n");
  }

  /* Print screen (make your terminal font very small to see this) */
  /* for (row = 0; row < 210; row++) {
    for (col = 0; col < 160; col++)
      fprintf (stderr, "%2x ", this_observation->intArray[128+col + row*160]);
    fprintf (stderr, "\n");
  } */

  fprintf (stderr, "\n");

  /* In a real action you might want to store the last observation and last action*/
	replaceRLStruct(&this_action, &last_action);
	replaceRLStruct(this_observation, last_observation);
	
	return &this_action;
}

void agent_end(double reward) {
	clearRLStruct(&last_action);
	clearRLStruct(last_observation);
}

void agent_cleanup() {
	clearRLStruct(&this_action);
	clearRLStruct(&last_action);
	freeRLStructPointer(last_observation);
}

const char* agent_message(const char* inMessage) {
	if(strcmp(inMessage,"what is your name?")==0)
		return "my name is skeleton_agent!";

	return "I don't know how to respond to your message";
}
