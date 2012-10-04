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
#include <config.h>
/* $Id: elements.c,v 1.1 2006/06/11 07:13:24 urchlay Exp $
 */
char *elements_id = "$Id: elements.c,v 1.1 2006/06/11 07:13:24 urchlay Exp $";
#include <stdio.h>
#include <math.h>
#include "rsynth.h"
#include "phfeat.h"


Elm_t Elements[] = {
#include "Elements.def"
};

unsigned num_Elements = (sizeof(Elements) / sizeof(Elm_t));

char *Ep_name[nEparm] = {
    "fn", "f1", "f2", "f3",
    "b1", "b2", "b3", "pn",
    "a2", "a3", "a4",
    "a5", "a6", "ab", "av",
    "avc", "asp", "af"
};

speaker_t *
rsynth_speaker(float F0Hz, float gain, Elm_t * e)
{
    static speaker_t pars;
    // memset(&pars,-1,sizeof(pars));
    pars.F0Hz = F0Hz;

    /* Quasi fixed parameters */
    pars.Gain0 = gain;
    pars.F4hz = 3900;
    pars.B4hz = 400;
    pars.F5hz = 4700;
    pars.B5hz = 150;
    pars.F6hz = 4900;
    pars.B6hz = 150;

    pars.B4phz = 500;
    pars.B5phz = 600;
    pars.B6phz = 800;

    pars.BNhz = 500;

    /* Set the _fixed_ nasal pole to nasal zero frequency of the 0th element
       (which should NOT be a nasal!) as a typical example of the zero
       we wish to cancel

     */
    pars.FNPhz = (long) e->p[fn].stdy;
    return &pars;
}
