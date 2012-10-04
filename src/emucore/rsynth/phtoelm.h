/* $Id: phtoelm.h,v 1.2 2006/06/11 21:49:09 stephena Exp $
*/
#ifndef __PHTOELM_H
#define __PHTOELM_H

#ifdef __cplusplus
extern "C" {
#endif

struct rsynth_s;
extern unsigned phone_to_elm (char *s, int n, darray_ptr elm, darray_ptr f0);
extern void say_pho(struct rsynth_s *rsynth, const char *path, int dodur,char *phoneset);

#ifdef __cplusplus
}
#endif

#endif
