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
#include "config.h"

/* $Id: phtoelm.c,v 1.2 2006/06/12 14:12:52 stephena Exp $
 */
char *phtoelm_id = "$Id: phtoelm.c,v 1.2 2006/06/12 14:12:52 stephena Exp $";
#include <stdio.h>
#include <ctype.h>
#include <stdlib.h>
#include <math.h>
#include <stdarg.h>
#include "useconfig.h"
#include "rsynth.h"
#include "trie.h"
#include "phfeat.h"


trie_ptr phtoelm = NULL;

static Elm_ptr find_elm(char *s);

static Elm_ptr
find_elm(char *s)
{
    Elm_ptr e = Elements;
    while (e < Elements + num_Elements) {
	if (!strcmp(s, e->name)) {
	    return e;
	}
	e++;
    }
    return NULL;
}

#define COMMENT(s)

static void
enter(char *p, ...)
{
    va_list ap;
    char *s;
    char buf[20];
    char *x = buf + 1;
    va_start(ap, p);
    while ((s = va_arg(ap, char *))) {
	Elm_ptr e = find_elm(s);
	if (e)
	    *x++ = (e - Elements);
	else {
	    fprintf(stderr, "Cannot find %s for [%s]\n", s, p);
	}
    }
    va_end(ap);
    buf[0] = (x - buf) - 1;
    x = (char *) malloc(buf[0] + 1);
    memcpy(x, buf, buf[0] + 1);
    trie_insert(&phtoelm, p, x);
}

static void
enter_phonemes(void)
{
#include "phtoelm.def"
}

trie_ptr
enter_trans(char *trans, int verbose)
{
    trie_ptr table = NULL;
    FILE *f = fopen(trans, "r");
    if (f) {
	char buf[1024];
	char *s;
	if (verbose)
	    fprintf(stderr, "Reading %s\n", trans);
	while ((s = fgets(buf, sizeof(buf), f))) {
	    char *p;
	    char *x;
	    while (isspace((unsigned) *s))
		s++;
	    p = s;
	    while (*s && !isspace((unsigned) *s))
		s++;
	    while (isspace((unsigned) *s))
		*s++ = '\0';
	    x = (char *) trie_lookup(&phtoelm, &s);
	    while (isspace(*s))
		s++;
	    if (*s) {
		fprintf(stderr, "%s does not map (leaves %s)\n", p, s);
	    }
	    else {
		trie_insert(&table, p, x);
	    }
	}
	fclose(f);
    }
    else {
	perror(trans);
    }
    return table;
}


#if 1
#define StressDur(e,s,l) ((e->ud + (e->du - e->ud) * s / 3)*speed)
#else
#define StressDur(e,s,l) ((void)(s),(l ? e->du*speed : (e->du + e->ud)/2)*speed)
#endif

static float
decline_f0(float F0Hz, darray_ptr f0, float f, unsigned t)
{
    darray_float(f0, t);
    /* Magic constant - ugh */
    f -= 0.12 * t;
    /* Do not drop forever */
    if (f < 0.7 * F0Hz)
	f = 0.7 * F0Hz;
    return darray_float(f0, f);
}

static unsigned
phone_to_elm(rsynth_t * rsynth, int n, char *phone, darray_ptr elm,
	     darray_ptr f0)
{
    float F0Hz = rsynth->speaker->F0Hz;
    float speed = rsynth->speed;
    int stress = 0;
    int seen_vowel = 0;
    int islong = 0;
    char *s = phone;
    unsigned t = 0;
    unsigned f0t = 0;
    char *limit = s + n;
    float f = darray_float(f0, F0Hz * 1.1F);
    if (!phtoelm)
	enter_phonemes();
    while (s < limit && *s) {
	char *e = (char *) trie_lookup(&phtoelm, &s);
	if (e) {
	    int n = *e++;
	    while (n-- > 0) {
		int x = *e++;
		Elm_ptr p = &Elements[x];
		darray_append(elm, x);
		/* StressDur works because only vowels have ud != du,
		   and we set stress just before a vowel
		 */
		t += darray_append(elm,
				   (int) (StressDur(p, stress, islong)));
		if (p->feat & vwl) {
		    seen_vowel = 1;
		}
		else if (seen_vowel) {
		    stress = 0;
		}
	    }
	}
	else {
	    char ch = *s++;
	    switch (ch) {
	    case '\'':		/* Primary stress */
		stress++;
	    case ',':		/* Secondary stress */
		stress++;
	    case '+':		/* Tertiary stress */
		stress++;
		if (stress > 3)
		    stress = 3;
		seen_vowel = 0;
		/* f0 has been declining since f0t */
		f = decline_f0(F0Hz, f0, f, t - f0t);
		f0t = t;
		/* Now stress pulse pushes f0 up "instantly" */
		darray_float(f0, 0);
		darray_float(f0, f + F0Hz * stress * 0.02);
		break;
	    case '-':		/* hyphen in input */
		break;
	    case ':':		/* Length mark */
		islong = 1;
		break;
	    default:
		fprintf(stderr, "Ignoring %c in '%.*s'\n", ch, n, phone);
		break;
	    }
	}
    }
    /* Add final decline to f0 contour */
    decline_f0(F0Hz, f0, f, t - f0t);
    return t;
}

void
rsynth_phones(rsynth_t * rsynth, char *phone, int len)
{
    darray_t elm;
    darray_t f0;
    unsigned frames;
    darray_init(&elm, sizeof(char), len);
    darray_init(&f0, sizeof(float), len);
    if ((frames = phone_to_elm(rsynth, len, phone, &elm, &f0))) {
	if (rsynth_verbose(rsynth))
	    fprintf(stderr, "[%.*s]\n", len, phone);
	rsynth_flush(rsynth,
	             rsynth_interpolate(rsynth,
					(unsigned char *) darray_find(&elm, 0),elm.items,
						(float *) darray_find(&f0,0),
						f0.items));
    }
    darray_free(&f0);
    darray_free(&elm);
}

void
rsynth_pho(rsynth_t * rsynth, const char *path, int dodur, char *trans)
{
    int verbose = rsynth_verbose(rsynth);
    FILE *f = fopen(path, "r");
    trie_ptr table;
    if (!phtoelm)
	enter_phonemes();
    if (trans && *trans && strcmp(trans, "sampa"))
	table = enter_trans(trans, verbose);
    else
	table = phtoelm;
    if (f) {
	char buffer[1024];
	char *s;
	darray_t elm;
	unsigned t = 0;
	float f0a[1024] = { rsynth->speaker->F0Hz };
	unsigned nf0 = 0;	/* index of last f0 value */
	unsigned f0t = 0;	/* time of last f0 value */
	darray_init(&elm, sizeof(char), 1024);
	if (verbose) {
	    fprintf(stderr, "Frame is %.3gms\n",
		    rsynth->samples_frame * 1000.0 / rsynth->sr);
	}
	while ((s = fgets(buffer, sizeof(buffer), f))) {
	    /* skip leading space - should not be any but ... */
	    while (isspace((unsigned) *s))
		s++;
	    if (*s && *s != ';') {
		/* Not a comment */
		char *ps = s;
		char *e = (char *) trie_lookup(&table, &s);
		if (*s == ':')
		    s++;
		if (e && isspace((unsigned) *s)) {
		    char *pe = s;
		    unsigned pt = 0;
		    int n = *e++;
		    int i;
		    double ms = strtod(s, &s);
		    float frames =
			ms * (rsynth->sr / rsynth->samples_frame) / 1000;
		    float edur = 0;
		    float estp = 0;
		    int nstp = 0;
		    float delta = 0;
		    for (i = 0; i < n; i++) {
			int x = e[i];
			Elm_ptr p = &Elements[x];
			if (!p->du || p->feat & stp)
			    estp += p->ud;
			else {
			    edur += p->ud;
			    nstp++;
			}
		    }
		    /* Stops don't change length */
		    frames -= estp;
		    delta = frames - edur;
#if 0
		    /* FIXME - revisit the rounding process */
		    if (verbose)
			fprintf(stderr,
				"'%.*s' %gms %d elem %g frames vs %g nat d=%g) %d stops\n",
				(pe - ps), ps, ms, n, frames, edur, delta,
				n - nstp);
#endif
		    for (i = 0; i < n; i++) {
			int x = e[i];
			Elm_ptr p = &Elements[x];
			darray_append(&elm, x);
			if (!p->du || p->feat & stp)
			    pt += darray_append(&elm, p->ud);
			else {
			    if (dodur) {
				float share =
				    (nstp >
				     1) ? rint(delta * (p->ud -
							1) / (edur -
							      nstp)) :
				    delta;
#if 0
				fprintf(stderr,
					"%s d=%d vs nstp=%g delta=%g take=%g\n",
					p->name, p->ud, edur, delta,
					share);
#endif
				edur -= p->ud;
				delta -= share;
				nstp--;
				pt +=
				    darray_append(&elm,
						  (int) (p->ud + share));
			    }
			    else
				pt += darray_append(&elm, p->du);
			}
		    }
		    /* Now have elements entered and duration of phone computed */
		    if (verbose && dodur) {
			float got =
			    1.0 * pt * rsynth->samples_frame / rsynth->sr *
			    1000;
			if (fabs(got - ms) > 0.5) {
			    fprintf(stderr,
				    "'%.*s' want=%gms got=%.3g (%+3.0f%%)\n",
				    (pe - ps), ps, ms, got,
				    100 * (got - ms) / ms);
			}
		    }
		    while (isspace((unsigned) *s))
			s++;
		    while (*s) {
			float percent = strtod(s, &s);
			float f0 = strtod(s, &s);
			unsigned nt =
			    (unsigned) (t + (percent * pt / 100));
			if (nt > f0t) {
			    /* time has advanced */
			    f0a[++nf0] = (nt - f0t);
			    f0a[++nf0] = f0;
			    f0t = nt;
			}
			else {
			    /* same time - change target inplace */
			    f0a[nf0] = f0;
			}
			while (isspace((unsigned) *s))
			    s++;
		    }
		    t += pt;
		}
		else {
		    fputs(buffer, stderr);
		    fprintf(stderr, "Unknown phone:%s", ps);
		}
	    }
	}
	fclose(f);
	if (t) {
	    float f0 = f0a[nf0++];
	    if (f0t < t) {
		f0a[nf0++] = t - f0t;
		f0a[nf0++] = f0;
	    }
	    rsynth_flush(rsynth,
	        rsynth_interpolate(rsynth,
			    (unsigned char *) darray_find(&elm,	0),
    	    	    	    elm.items, f0a, nf0));
	}
    }
    else {
	perror(path);
    }
}

void
rsynth_term(rsynth_t * rsynth)
{
    if (rsynth) {
	rsynth_flush(rsynth, 0);
	trie_free(&phtoelm, &free);
#ifdef DO_RANGE_CHECKS
	fprintf(stderr, "Max range %g @ %s:%d\n", range_max, __FILE__,
		range_ln);
#endif
	if (rsynth->voice_file)
	    fclose(rsynth->voice_file);
	if (rsynth->parm_file)
	    fclose(rsynth->parm_file);
	free(rsynth->pvt);
	free(rsynth);
    }
}
