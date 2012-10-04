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
/* $Id: darray.h,v 1.2 2006/06/11 21:49:07 stephena Exp $
*/
#if !defined(DARRAY_H)
#define DARRAY_H
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
 {char     *data;          /* the items */
  unsigned items;          /* number of slots used */
  unsigned alloc;          /* number of slots allocated */
  unsigned short esize;    /* size of items */
  unsigned short get;      /* number to get */
 } darray_t, *darray_ptr;

/* return pointer to nth item */
extern void *Darray_find(darray_t *a,unsigned n);
/* delete nth item */
extern int darray_delete(darray_t *a,unsigned n);
extern void darray_free(darray_t *a);
extern int darray_append(darray_t *p, int ch);
extern void darray_cat(void *da, char *s);

#if defined(__GNUC__)
static __inline__ void darray_init(darray_t *a,unsigned size,unsigned get)
{
 a->esize = size;
 a->get   = get;
 a->items = a->alloc = 0;
 a->data = NULL;
}

static __inline__ void *darray_find(darray_t *a,unsigned n)
{
 if (n < a->alloc && n < a->items)
  return (void *) (a->data + n * a->esize);
 return Darray_find(a,n);
}

static inline float
darray_float(darray_ptr f0, float f)
{
    float *fp = (float *) darray_find(f0, f0->items);
    *fp = f;
    return f;
}

static inline short
darray_short(darray_ptr f0, short f)
{
    short *fp = (short *) darray_find(f0, f0->items);
    *fp = f;
    return f;
}

#else

extern float darray_float(darray_ptr f0, float f);
extern short darray_short(darray_ptr f0, short f);

#define darray_init(a,sz,gt) \
 ((a)->esize = (sz), (a)->get = (gt), (a)->items = (a)->alloc = 0, (a)->data = NULL)

#define darray_find(a,n) \
 (((n) < (a)->alloc && (n) < (a)->items) \
   ? (void *) ((a)->data + (n) * (a)->esize)  \
   : Darray_find(a,n))

#endif 

extern int darray_fget(FILE * f, darray_ptr p);

#ifdef __cplusplus
}
#endif

#endif /* DARRAY_H */
