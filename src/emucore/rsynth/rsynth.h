/*
    Copyright (c) 1994,2001-2003 Nick Ing-Simmons. All rights reserved.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
    MA 02111-1307, USA

*/
#ifndef RSYNTH_H
#define RSYNTH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "darray.h"
#include "trie.h"

typedef struct
 {float stdy;	    /* steady state value of paramter */
  char  prop;       /* Percentage of 'stdy' to add to adjacent element */
  char  ed;         /* External Duration of transition */
  char  id;         /* Internal Duration of transition */
  char  rk;         /* Rank of this element for transition dominance */
 } interp_t, *interp_ptr;

/*
fn      Nasal zero freq in Hz,           248 to  528
f1      First formant freq in Hz,        200 to 1300
f2      Second formant freq in Hz,       550 to 3000
f3      Third formant freq in Hz,        1200 to 4999
b1      First formant bw in Hz,          40 to 1000
b2      Second formant bw in Hz,         40 to 1000
b3      Third formant bw in Hz,          40 to 1000
pn  	Proportion of nasal,             0 to   1
a2  	Amp of F2 frication in dB,       0 to   57
a3  	Amp of F3 frication in dB,       0 to   50
a4  	Amp of F4 frication in dB,       0 to   46
a5  	Amp of F5 frication in dB,       0 to   40
a6  	Amp of F6 frication in dB,       0 to   43
ab  	Amp of bypass fric. in dB,       0 to   48
av      Amp of voicing in dB,            0 to   70
avc     Amp of lf voice,                 0 to   60
asp     Amp of aspiration in dB,         0 to   34
af      Amp of frication in dB,          0 to   60
*/

enum Eparm_e
 {
  fn, f1, f2, f3, b1, b2, b3, pn, a2, a3, a4, a5, a6, ab, av, avc, asp, af,
  nEparm
 };

extern char *Ep_name[nEparm];

typedef struct Elm_s
 {
  char *name;	    	    /* name of element */
  char rk;  	    	    /* Rank of element (obsolete) */
  char du;                  /* Normal duration */
  char ud;                  /* Unstressed duration */
  char *unicode;   /* UTF-8 for phonetic symbols */
  char  *sampa;             /* SAMPA string for phonetic symbol */
  long  feat;               /* features of the phone */
  interp_t p[nEparm];       /* Table of parameter values */
 } Elm_t, *Elm_ptr;

extern Elm_t Elements[];
extern unsigned num_Elements;

typedef struct
 {
  float Gain0;             /* Overall gain, 60 dB is unity,    0 to   60       */
  float F0Hz;              /* Typical voicing frequency */
  float F4hz;              /* Fourth formant freq in Hz,       1200 to 4999    */
  float B4hz;              /* Fourth formant bw in Hz,         40 to 1000      */
  float F5hz;              /* Fifth formant freq in Hz,        1200 to 4999    */
  float B5hz;              /* Fifth formant bw in Hz,          40 to 1000      */
  float F6hz;              /* Sixth formant freq in Hz,        1200 to 4999    */
  float B6hz;              /* Sixth formant bw in Hz,          40 to 2000      */
  float FNPhz;             /* Nasal pole freq in Hz,           248 to  528     */
  float BNhz;              /* Nasal pole/zero bw in Hz,        40 to 1000      */
  float B4phz;             /* Par. 4th formant bw in Hz,       40 to 1000      */
  float B5phz;             /* Par. 5th formant bw in Hz,       40 to 1000      */
  float B6phz;             /* Par. 6th formant bw in Hz,       40 to 2000      */
 } speaker_t;


typedef struct rsynth_s rsynth_t;
typedef void *rsynth_sample_p(void *user_data,float sample, unsigned nsamp, rsynth_t *rsynth);
typedef void *rsynth_flush_p(void *user_data,unsigned nsamp, rsynth_t *rsynth);

struct rsynth_s
{
  long flags;                   /* Various flag bits */
  long sr;                  	/* sample rate */
  long samples_frame;       	/* Number of samples in a frame */
  speaker_t *speaker;       	/* Current speaker (voice) characteristics */
  float *ep;                 	/* Paramters for current frame */
  FILE *voice_file;         	/* File to print voicing waveforms */
  FILE *parm_file;         	/* File to print parameter values */
  rsynth_sample_p *sample_p;    /* sample handler */
  rsynth_flush_p  *flush_p;     /* sample handler */
  void *user_data;              /* Argument to handlers */
  float speed;	    	    	/* element duration multiplier */
  float smooth;     	    	/* smoothing "filter" coefficent for parameters */
  struct rsynth_private *pvt;   /* Private data to backend synth */
};

#define RSYNTH_VERBOSE   (1L << 0)
#define RSYNTH_ETRACE    (1L << 1)
#define RSYNTH_MONOTONE  (1L << 2)
#define RSYNTH_F0TRACE   (1L << 3)

extern speaker_t *rsynth_speaker(float F0Hz, float gain, Elm_t *elements);

extern rsynth_t *rsynth_init(long samrate,
                             float ms_per_frame,
			     speaker_t *speaker,
			     rsynth_sample_p *sample_p,
			     rsynth_flush_p  *flush_p,
			     void *user_data);
			
/* Simple top level */
extern void rsynth_phones(struct rsynth_s *rsynth, char *s, int len);

/* Synth from "mbrola" style .pho file */
extern void rsynth_pho(struct rsynth_s *rsynth, const char *path, int dodur,char *phoneset);

			
/* Middle level - interpolate parameters of element sequence */			
extern unsigned rsynth_interpolate(rsynth_t *rsynth,
                       unsigned char *elm, unsigned nelm,
                       float *f0, unsigned nf0);
		
extern void rsynth_flush(rsynth_t *rsynth,unsigned nsamp);		
		
/* Bottom level - generate a frame based on paramters */		
extern long rsynth_frame(rsynth_t *rsynth, float F0Hz, float *frame, const char *name);

extern void rsynth_term(rsynth_t *rsynth);

#define rsynth_verbose(rsynth) ((!rsynth) || (rsynth)->flags & RSYNTH_VERBOSE)

#ifdef __cplusplus
}
#endif


#endif
