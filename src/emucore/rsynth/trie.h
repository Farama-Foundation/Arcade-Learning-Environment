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
/* $Id: trie.h,v 1.1 2006/06/11 07:13:27 urchlay Exp $
*/
#ifndef TRIE_H
#define TRIE_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct trie_s *trie_ptr;

extern void trie_insert(trie_ptr *r,char *s,void *value);
extern void *trie_lookup(trie_ptr *r,char **sp);
extern void trie_free(trie_ptr *r,void (*func)(void *));

#ifdef __cplusplus
}
#endif


#endif /* TRIE_H */
