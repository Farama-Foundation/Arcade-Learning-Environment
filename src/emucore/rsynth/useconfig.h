#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>
#endif

#if STDC_HEADERS
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#else          
#ifndef HAVE_STRCHR          
#define strchr index          
#define strrchr rindex          
#endif          
char *strchr (), *strrchr ();
#ifndef HAVE_MEMCPY          
#define memcpy(d, s, n) bcopy ((s), (d), (n))          
#endif
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif
#endif /* STDC_HEADERS */

#if HAVE_UNISTD_H
#include <unistd.h>
#endif

#if HAVE_LIBC_H
/* From NeXT stuff */
#include <libc.h>
#endif

