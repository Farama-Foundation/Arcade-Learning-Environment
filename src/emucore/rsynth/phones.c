/*
    Copyright (c) 1994,2001-2002 Nick Ing-Simmons. All rights reserved.
 
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
#include <stdio.h>
#include "phones.h"

#if 0
#ifdef __STDC__
#define PHONE(nm,br,am,ex) #nm,
#else
#define PHONE(nm,br,am,ex) "nm",
#endif
#endif

#define PHONE(nm,st,br,am,ex) st

char *ph_name[] =
{" ",
#include "phones.def"
 NULL};
#undef PHONE

#define PHONE(nm,st,br,am,ex) br
char *ph_br[] =
{" ",
#include "phones.def"
 NULL};
#undef PHONE

#define PHONE(nm,st,br,am,ex) am
char *ph_am[] =
{" ",
#include "phones.def"
 NULL};
#undef PHONE
