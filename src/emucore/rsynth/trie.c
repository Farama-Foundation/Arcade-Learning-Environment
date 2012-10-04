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

#include "config.h"
/* $Id: trie.c,v 1.1 2006/06/11 07:13:27 urchlay Exp $
 */
char *trie_id = "$Id: trie.c,v 1.1 2006/06/11 07:13:27 urchlay Exp $";
#include "useconfig.h"
#include <stdio.h>
#include "trie.h"

struct trie_s {
    struct trie_s *otherwise;
    struct trie_s *more;
    void *value;
    char ch;
};

void
trie_free(trie_ptr * r, void (*func) (void *))
{
    trie_ptr p;
    while ((p = *r)) {
	trie_free(&p->more, func);
	*r = p->otherwise;
	if (func)
	    (*func) (p->value);
	free(p);
    }
}

void
trie_insert(trie_ptr * r, char *s, void *value)
{
    trie_ptr p = NULL;
    char ch;
    while ((ch = *s++)) {
	while ((p = *r)) {
	    if (p->ch == ch)
		break;
	    else
		r = &p->otherwise;
	}
	if (!p) {
	    p = (trie_ptr) malloc(sizeof(*p));
	    memset(p, 0, sizeof(*p));
	    p->ch = ch;
	    *r = p;
	}
	r = &p->more;
    }
    p->value = value;
}

void *
trie_lookup(trie_ptr * r, char **sp)
{
    char *s = *sp;
    char *value = NULL;
    char ch;
    while ((ch = *s)) {
	trie_ptr *l = r;
	trie_ptr p;
	while ((p = *l)) {
	    if (p->ch == ch)
		break;
	    else
		l = &p->otherwise;
	}
	if (p) {
	    *l = p->otherwise;
	    p->otherwise = *r;
	    *r = p;
	    r = &p->more;
	    value = (char *) p->value;
	    s++;
	}
	else
	    break;
    }
    *sp = s;
    return value;
}
