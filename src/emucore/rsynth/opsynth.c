/*
    Copyright (c) 2002-2004 Nick Ing-Simmons. All rights reserved.

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
#include <config.h>
#include <useconfig.h>
#include <stdio.h>
#define __USE_ISOC99 1		/* For fpclassify() etc. */
#include <math.h>
#ifndef PI
#define PI            3.1415927
#endif
#include "rsynth.h"

#define SYNC_CASCADE

#define PVT(x) rsynth->pvt->x
#define RES(x) &PVT(x),#x


#if defined(__linux__)
#define FPCHECK(x) do {                 \
 if (fpclassify(x) < FP_ZERO)           \
  {                                     \
   fprintf(stderr, #x "=%g (%s:%d)\n",x,__FILE__,__LINE__);        \
   abort();                             \
  }                                     \
} while (0)
#else
#define FPCHECK(x)
#endif

#ifdef DO_RANGE_CHECKS

static float range_max = 0.0;	/* largest value seen */
static int range_ln = 0;	/* Line number it occurred */

#define RGCHECK(x) do {							\
 FPCHECK(x);								\
 if (abs(x) > range_max)						\
  {									\
   range_max = abs(x);							\
   if (range_max > 32767)						\
    fprintf(stdout, #x "=%g (%s:%d)\n",range_max,__FILE__,__LINE__);	\
   range_ln  = __LINE__;						\
  }									\
} while (0)
#else
#define RGCHECK(x)
#endif



typedef struct {
    float a;
    float b;
    float c;
    float p1;
    float p2;
} resonator_t, *resonator_ptr;

typedef struct {
    float a;
    float b;
    float p1;
} lowpass_t, *lowpass_ptr;

struct rsynth_private {
    /* Voicing source */
    long nper;			/* Current loc in voicing period 4*sample_rate */
    long T0;			/* Fundamental period in output samples times 4 */
    long nopen;			/* Number of samples in open phase of period  */
    float F0Hz;

    float amp_av;		/* ep[av] converted to linear gain  */
    float amp_bypass;		/* ep[ab] converted to linear gain  */
    float amp_asp;		/* ep[asp] converted to linear gain  */
    float amp_af;		/* ep[af] converted to linear gain  */
    float amp_avc;		/* AVC converted to linear gain  */
    float amp_turb;		/* Turbuleance in voicing */

    unsigned long seed;
    unsigned long ns;
    long clip_max;
    unsigned usrsamp;

    resonator_t rgl;
    resonator_t rnz;
    resonator_t rnpc;
    resonator_t r5c;
    resonator_t rsc;
    resonator_t r4c;
    resonator_t r3c;
    resonator_t r2c;
    resonator_t r1c;
    resonator_t r6p;
    resonator_t r5p;
    resonator_t r4p;
    resonator_t r3p;
    resonator_t r2p;
    resonator_t rout;
};

/* Convert formant freqencies and bandwidth into
   resonator difference equation coefficents
 */
static void
set_pole_fbw(long sr, float f, float bw, resonator_ptr rp, char *name,
	     int casc)
{
    float minus_pi_t = -PI / sr;
    float two_pi_t = -2.0 * minus_pi_t;

    /* Attempt to be clever and automatically adjust resonator presence based
       on sample rate -it does not work very well.
       So cascade path has its own conditional to skip r5c
     */
    if (2 * f - bw <= sr) {
	double arg;
	float r;
	if (2 * (f + bw) > sr) {
	    /* This is a little dubious - keep lower skirt in place and
	       move centre frequency so upper skirt hits Nyquist freq
	     */
	    float low = f - bw;
#if 0
	    fprintf(stderr, "Adj %s %ld/%ld @ %ld", name, f, bw, sr);
#endif
	    f = (sr / 2 + low) / 2;
	    bw = f - low;
#if 0
	    fprintf(stderr, "=> %ld/%ld\n", f, bw);
#endif
	}
	arg = minus_pi_t * bw;
	r = exp(arg);		/* Let r  =  exp(-pi bw t) */
	rp->c = -(r * r);	/* Let c  =  -r**2 */
	arg = two_pi_t * f;
	rp->b = r * cos(arg) * 2.0;	/* Let b = r * 2*cos(2 pi f t) */
	rp->a = 1.0 - rp->b - rp->c;	/* Let a = 1.0 - b - c */
    }
    else {
	/* lower skirt of resonator exceeds Nyquist limit - make it a no-op,
	   i.e. pass if cascade, zero if parallel
	 */
#if 0
	fprintf(stderr, "Skip %s %ld/%ld @ %ld\n", name, f, bw, sr);
#endif
	rp->a = casc;
	rp->b = 0.0;
	rp->c = 0.0;
    }
}

/* Convienience function for setting parallel resonators with gain */
static void
set_pole_fbwg(long sr, float f, float bw, resonator_ptr rp, char *name,
	      float gain, int casc)
{
    set_pole_fbw(sr, f, bw, rp, name, casc);
    rp->a *= gain;
}

/* Convert formant freqencies and bandwidth into
 *      anti-resonator difference equation constants
 */
static void
set_zero_fbw(long sr, float f, float bw, resonator_ptr rp, char *name)
{
    /* First compute ordinary resonator coefficients */
    set_pole_fbw(sr, f, bw, rp, name, 1);
    /* Now convert to antiresonator coefficients */
    rp->a = 1.0 / rp->a;	/* a'=  1/a */
    rp->b *= -rp->a;		/* b'= -b/a */
    rp->c *= -rp->a;		/* c'= -c/a */
}


/* Generic resonator function */
static float
resonator(resonator_ptr r, char *name, float input)
{
    register float x = r->a * input + r->b * r->p1 + r->c * r->p2;
    FPCHECK(input);
    FPCHECK(r->a);
    FPCHECK(r->b);
    FPCHECK(r->c);
    r->p2 = r->p1;
    r->p1 = x;
    return x;
}

/* Generic anti-resonator function
   Same as resonator except that a,b,c need to be set with set_zero_fbw()
   and we save inputs in p1/p2 rather than outputs.
   There is currently only one of these - "rnz"
 */
/*  Output = (rnz.a * input) + (rnz.b * oldin1) + (rnz.c * oldin2) */

static float
antiresonator(resonator_ptr r, char *name, float input)
{
    register float x = r->a * input + r->b * r->p1 + r->c * r->p2;
    FPCHECK(input);
    FPCHECK(r->a);
    FPCHECK(r->b);
    FPCHECK(r->c);
    r->p2 = r->p1;
    r->p1 = input;
    return x;
}


/* Convert from decibels to a linear scale factor */
static float
DBtoLIN(float dB)
{
    if (dB > 0) {
	float val = 32768 * pow(10.0, (dB - 87) / 20 - 3);
	FPCHECK(val);
	return val;
    }
    return 0.0;
}

static void
set_cascade(rsynth_t * rsynth)
{
    long sr = rsynth->sr;
    /* Set coeficients of nasal resonator and zero antiresonator */
    set_pole_fbw(sr, rsynth->speaker->FNPhz, rsynth->speaker->BNhz,
		 RES(rnpc), 1);
    set_zero_fbw(sr, rsynth->ep[fn], rsynth->speaker->BNhz, RES(rnz));
    /* Rest of cascade path */
    set_pole_fbw(sr, 3500, 1800, RES(rsc), 1);
    set_pole_fbw(sr, rsynth->speaker->F5hz, rsynth->speaker->B5hz,
		 RES(r5c), 1);
    set_pole_fbw(sr, rsynth->speaker->F4hz, rsynth->speaker->B4hz,
		 RES(r4c), 1);
    set_pole_fbw(sr, rsynth->ep[f3], rsynth->ep[b3], RES(r3c), 1);
    set_pole_fbw(sr, rsynth->ep[f2], rsynth->ep[b2], RES(r2c), 1);
    set_pole_fbw(sr, rsynth->ep[f1], rsynth->ep[b1], RES(r1c), 1);
}

static void
pitch_sync(rsynth_t * rsynth)
{
    float F0Hz = PVT(F0Hz);

    if (rsynth->ep[av] > 0 || rsynth->ep[avc] > 0) {
	PVT(T0) = (long) ((4 * rsynth->sr) / F0Hz);	/* Period in samp*4 */
	PVT(amp_av) = DBtoLIN(rsynth->ep[av]);	/* Voice amplitude */
	PVT(amp_avc) = DBtoLIN(rsynth->ep[avc]);	/* Voice-bar amplitude */
	PVT(amp_turb) = PVT(amp_avc) * 0.1;	/* Breathiness of voicing waveform */


	/* Duration of period before amplitude modulation */
	/* if voiced then nopen is 1/3rd the period */
	PVT(nopen) = PVT(T0) / 3;
	/* Klatt's original has code to jitter T0 by a skew */
    }
    else {
	PVT(T0) = 4;		/* Default for f0 undefined */
	PVT(nopen) = PVT(T0);
	PVT(amp_av) = 0.0;
	PVT(amp_avc) = 0.0;
    }

    /* Reset these pars pitch synchronously or at update rate if f0=0 */

    if ((PVT(T0) != 4) || (PVT(ns) == 0)) {
	if (rsynth->voice_file) {
	    fprintf(rsynth->voice_file, "# pitch sync T0=%ld\n", PVT(T0));
	}
	/* rgl is used to smooth voice to feed cascade during voice-bars
	   of plosives etc. - high frequencies are much reduced
	   but try and keep some 2nd harmonic by setting cuttoff to twice f0.
	 */
	set_pole_fbw(rsynth->sr, 0L, (long) (2 * F0Hz), RES(rgl), 1);


	/* Holmes also says that when glotis is open BW of formants
	   is wider due to damping of sub-glottal airway
	   We are not modelling this yet.
	 */

#ifdef SYNC_CASCADE
	/* One theory was that some of noises were due to setting parameters
	   and that doing cascade (voice) part sync would help - it didn't
	 */
	set_cascade(rsynth);
#endif
    }
}

static float
gen_voice(rsynth_t * rsynth, float noise)
{
    int i;
    float voice;
    /* Generate voice waveform at sr*4 to give better resolution
       of T0 and hence period of f0
       The sr*4 was for 10kHz system - using impulse train
       as voicing source.
       With interpolated voice scheme, and without the
       lowpass filter used to cancel output 1st difference
       this loop seems to be pointless.
     */
    for (i = 0; i < 4; i++) {
	float alpha;
	const float amp = 4096.0;
	if (PVT(nper) >= PVT(T0)) {
	    PVT(nper) = 0;
	    pitch_sync(rsynth);
	}
	/* Voice source shape is linear ramp up for 1/3 of
	   period, then instant drop to baseline and
	   a parabolic loop below baseline for 2nd 2/3rds
	   of period. Shape developed by doing LPC analysis
	   of sample vowels and inverse filtering raw (un differenced)
	   sample data to approximate excitation of LPC filter.
	   May still be too "spiky" to be natural, and cosine arch
	   might be better than parabola.
	 */
	alpha = (float) PVT(nper) / PVT(T0);
	if (alpha <= 1.0 / 3) {
	    voice = 3 * amp * alpha;
	}
	else {
	    voice = amp * ((9 * alpha - 12) * alpha + 3);
	}

	/* The Klatt model had a low-passed voice term
	   which needs further analysis
	 */

	PVT(nper)++;
    }
    FPCHECK(voice);
    return voice;
}

float
gen_noise(rsynth_t * rsynth)
{
    float noise = 0.0;
    /* pseudo-ramdom is uniformly distributed so average 16 randoms
       to get approximatly Gaussian distribution
     */
    int i;
    for (i = 0; i < 16; i++) {
	/* Our own code like rand(), but portable.
	   Whole upper 31 bits of seed random
	   assumes 32-bit unsigned arithmetic
	   with untested code to handle larger.
	 */
	long nrand;
	PVT(seed) = PVT(seed) * 1664525 + 1;
	if (8 * sizeof(unsigned long) > 32)
	    PVT(seed) &= 0xFFFFFFFFL;
	/* Low have 31bits of random we want 14 MS bits as a signed value */
	/* Shift top bits of seed up to top of long then back down to LS 14 bits */
	/* Assumes 8 bits per sizeof unit i.e. a "byte" */
	nrand =
	    (((long) PVT(seed)) << (8 * sizeof(long) -
				    32)) >> (8 * sizeof(long) - 14);
	noise += nrand;
    }
    /* now divide by 2 - just to balance amplitide of noise with voice source */
    return noise / 2;
}

static void
setup_frame(rsynth_t * rsynth)
{
    long sr = rsynth->sr;
    float Gain0 = rsynth->speaker->Gain0 - 3;

#ifndef SYNC_CASCADE
    set_cascade(sr, frame);
#endif

    /* Set coefficients of parallel resonators, and amplitude of outputs */
    /* dB adjustment is to make 60db "full-on" for all parameters */

    set_pole_fbwg(sr, rsynth->ep[f2], rsynth->ep[b2], RES(r2p),
		  DBtoLIN(rsynth->ep[a2]), 0);

    set_pole_fbwg(sr, rsynth->ep[f3], rsynth->ep[b3], RES(r3p),
		  DBtoLIN(rsynth->ep[a3]), 0);

    set_pole_fbwg(sr, rsynth->speaker->F4hz, rsynth->speaker->B4phz,
		  RES(r4p), DBtoLIN(rsynth->ep[a4]), 0);

    set_pole_fbwg(sr, rsynth->speaker->F5hz, rsynth->speaker->B5phz,
		  RES(r5p), DBtoLIN(rsynth->ep[a5]), 0);

    set_pole_fbwg(sr, rsynth->speaker->F6hz, rsynth->speaker->B6phz,
		  RES(r6p), DBtoLIN(rsynth->ep[a6]), 0);

    PVT(amp_bypass) = DBtoLIN(rsynth->ep[ab]);
    PVT(amp_asp) = DBtoLIN(rsynth->ep[asp]);
    PVT(amp_af) = DBtoLIN(rsynth->ep[af]);


    /* fold overall gain into output resonator */
    if (Gain0 <= 0)
	Gain0 = 57;
    /* output low-pass filter - resonator with freq 0 and BW = samrate
     */
    set_pole_fbwg(sr, 0L, sr / 2, RES(rout), DBtoLIN(Gain0), 1);
}

float
rsynth_filter(rsynth_t * rsynth, float voice, float noise)
{
    RGCHECK(noise);
    RGCHECK(voice);

    /* Glottal source down cascade chain via nasal zero/pole and f1..f5 */
    voice = resonator(RES(rnpc), voice);
    RGCHECK(voice);
    voice = antiresonator(RES(rnz), voice);
    RGCHECK(voice);
    voice = resonator(RES(r1c), voice);
    RGCHECK(voice);
    voice = resonator(RES(r2c), voice);
    RGCHECK(voice);
    voice = resonator(RES(r3c), voice);
    RGCHECK(voice);
    voice = resonator(RES(r4c), voice);
    RGCHECK(voice);
    voice = resonator(RES(rsc), voice);
    RGCHECK(voice);
    if (rsynth->sr > 8000) {
	voice = resonator(RES(r5c), voice);
	RGCHECK(voice);
    }

    /* Now add the parallel parts - excited by frication, alternating in phase.
       The phase alternation is to mimimize "zeros" in spectrum by making use
       of the fact that phase skirts of two formants F(N) and F(N+1) will
       be opposite sign - so for trick to work we must do this in formant
       order
     */
    /* No f1 on parallel side */
    voice = resonator(RES(r2p), noise) - voice;
    RGCHECK(voice);
    voice = resonator(RES(r3p), noise) - voice;
    RGCHECK(voice);
    voice = resonator(RES(r4p), noise) - voice;
    RGCHECK(voice);
    voice = resonator(RES(r5p), noise) - voice;
    RGCHECK(voice);
    voice = resonator(RES(r6p), noise) - voice;
    RGCHECK(voice);
    voice = PVT(amp_bypass) * noise - voice;
    RGCHECK(voice);

    /* Final low-pass filter and gain */
    voice = resonator(RES(rout), voice);
    RGCHECK(voice);
    return voice;
}

long
rsynth_frame(rsynth_t * rsynth, float F0Hz, float *frame, const char *name)
{
    unsigned long es;
    rsynth->ep = frame;
    setup_frame(rsynth);
    if (rsynth->voice_file) {
	fprintf(rsynth->voice_file, "# voice lpvoc noise    out");
	fprintf(rsynth->voice_file, "\n");
	if (name) {
	    fprintf(rsynth->voice_file, "# %s\n", name);
	}
    }
    PVT(F0Hz) = F0Hz;

/*
 fprintf(stderr,"F0=%g T0=%ld @ %ld\n",0.1*F0hz10,T0,ns);
*/
    for (es = PVT(ns) + rsynth->samples_frame; PVT(ns) < es; PVT(ns)++) {
	float noise = gen_noise(rsynth);
	float voice = gen_voice(rsynth, noise);
	float lpvoice = resonator(RES(rgl), voice);
	if (PVT(nper) < PVT(nopen)) {
	    /* Add breathiness during glottal open phase - using noise
	       Note that amp_avc is further modified by amp_av below.
	     */
	    voice += PVT(amp_turb) * noise;
	}
	RGCHECK(noise);

	/* Original Klatt derived code had a 1st difference
	   which needed a compsenating soft low-pass filter.
	 */
	if (PVT(nper) < PVT(nopen)) {
	    /* less noise in 2nd half of glottal (open phase) */
	    /* Should that apply to frictation ? */
	    noise *= 0.5;
	}

	voice *= PVT(amp_av);
	/* Add in aspiration noise to glottal source */
	voice += (PVT(amp_asp) * noise);
#if 1
	/* Add in sinusoidal voice-bar element */
	voice += (PVT(amp_avc) * lpvoice);
#endif


	/* From now on noise is frictation source */
	noise *= PVT(amp_af);

	if (rsynth->voice_file) {
	    fprintf(rsynth->voice_file, "%6g %6g %6g", voice, lpvoice,
		    noise);
	}

	voice = rsynth_filter(rsynth, voice, noise);

	if (rsynth->voice_file) {
	    fprintf(rsynth->voice_file, " %6g\n", voice);
	}


	/* Original Klatt code did  1st difference of output - which
	   boosts high frequency.  Idea was based on the "lip radiation"
	   (terminating impedance) of the vocal-tract-as-tubes model.
	   As waveform is normally slowly varying
	   it also reduced amplitude considerably which meant earlier
	   stages needed a lot of dynamic range.
	   This synth attempts to get same overall response by folding
	   this high-pass element of transfer function into the excitation.
	   This not only keeps values more bounded but is computationally
	   simpler as high-pass is achieved by removing low-pass elements.
	   The 1st difference also exaggerated problems of an impulse
	   train as a voicing source and made sampling rate much more important.
	 */
	if (rsynth->sample_p)
	    rsynth->user_data =
		(*rsynth->sample_p) (rsynth->user_data, voice,
				     PVT(usrsamp)++, rsynth);
    }
    return PVT(usrsamp) + rsynth->samples_frame;
}

void
rsynth_flush(rsynth_t * rsynth, unsigned nsamp)
{
    if (!nsamp)
	nsamp = PVT(usrsamp);
    if (rsynth->flush_p)
	rsynth->user_data =
	    (*rsynth->flush_p) (rsynth->user_data, nsamp, rsynth);
    PVT(usrsamp) = 0;
}

/* This is here so we can keep rsynth_private private to this file */

rsynth_t *
rsynth_init(long sr, float ms_per_frame, speaker_t * speaker,
	    rsynth_sample_p * sample_p,
	    rsynth_flush_p * flush_p, void *user_data)
{
    rsynth_t *rsynth = (rsynth_t *) malloc(sizeof(rsynth_t));
    struct rsynth_private *pvt = (struct rsynth_private *)
	malloc(sizeof(struct rsynth_private));
    if (rsynth && pvt) {
	memset(rsynth, 0, sizeof(*rsynth));
	memset(pvt, 0, sizeof(*pvt));
	rsynth->pvt = pvt;
	PVT(seed) = 5;		/* Fixed staring value */
	rsynth->sr = sr;
	rsynth->samples_frame = (long) ((sr * ms_per_frame) / 1000);
	rsynth->speaker = speaker;
	rsynth->sample_p = sample_p;
	rsynth->flush_p = flush_p;
	rsynth->user_data = user_data;
	rsynth->smooth = 0.5F;
	rsynth->speed = 1.0F;
    }
    return rsynth;
}
