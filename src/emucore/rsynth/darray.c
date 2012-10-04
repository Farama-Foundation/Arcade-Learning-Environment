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
/* $Id: darray.c,v 1.1 2006/06/11 07:13:23 urchlay Exp $
 */
char *darray_id = "$Id: darray.c,v 1.1 2006/06/11 07:13:23 urchlay Exp $";
#include "useconfig.h"
#include "darray.h"

void
darray_free(darray_t * a)
{
    if (a->data) {
	free(a->data);
	a->data = NULL;
    }
    a->items = a->alloc = 0;
}

void *
Darray_find(darray_t * a, unsigned int n)
{
    if (n >= a->alloc || n >= a->items) {
	unsigned osize = a->items * a->esize;
	unsigned nsize;
	if (!a->esize)
	    abort();
	if (n >= a->alloc) {
	    unsigned add = (a->get) ? a->get : 1;
	    char *ndata = (char *) malloc(nsize = (n + add) * a->esize);
	    if (ndata) {
		if (osize)
		    memcpy(ndata, a->data, osize);
		if (a->data)
		    free(a->data);
		a->data = ndata;
		a->alloc = n + add;
	    }
	    else
		return NULL;
	}
	else
	    nsize = (n + 1) * a->esize;
	if (n >= a->items) {
	    memset(a->data + osize, 0, nsize - osize);
	    a->items = n + 1;
	}
    }
    return (void *) (a->data + n * a->esize);
}

int
darray_delete(darray_t * a, unsigned int n)
{
    char *p = (char *) darray_find(a, n);
    if (p) {
	if (a->items) {
	    a->items--;
	    while (n++ < a->items) {
		memcpy(p, p + a->esize, a->esize);
		p += a->esize;
	    }
	    memset(p, 0, a->esize);
	    return 1;
	}
	else
	    abort();
    }
    else
	return 0;
}

int
darray_append(darray_ptr p, int ch)
{
    char *s = (char *) darray_find(p, p->items);
    *s = ch;
    return ch;
}

void
darray_cat(void *arg, char *s)
{
    darray_ptr p = (darray_ptr) arg;
    char ch;
    while ((ch = *s++))
	darray_append(p, ch);
}

int
darray_fget(FILE * f, darray_ptr p)
{
    int ch;
    while ((ch = fgetc(f)) != EOF) {
	darray_append(p, ch);
	if (ch == '\n')
	    break;
    }
    darray_append(p, '\0');
    return p->items - 1;
}

#if !defined(__GNUC__)

float
darray_float(darray_ptr f0, float f)
{
    float *fp = (float *) darray_find(f0, f0->items);
    *fp = f;
    return f;
}

short
darray_short(darray_ptr f0, short f)
{
    short *fp = (short *) darray_find(f0, f0->items);
    *fp = f;
    return f;
}

#endif
