#ifndef OCL_H
#define OCL_H

#include "config.h"

#include <stdbool.h>
#ifdef __APPLE_CC__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "algorithm.h"

typedef struct __clState {
  cl_context context;
  cl_kernel kernel;
  cl_kernel *extra_kernels;
  size_t n_extra_kernels;
  cl_command_queue commandQueue;
  cl_program program;
  cl_mem outputBuffer;
  cl_mem CLbuffer0;
  cl_mem MidstateBuf;
  cl_mem padbuffer8;
  cl_mem BranchNonces;
  cl_mem Branch1Nonces;
  cl_mem Branch2Nonces;
  cl_mem Branch3Nonces;
  cl_mem Branch4Nonces;
  cl_mem Branch5Nonces;
  cl_mem Branch6Nonces;
  cl_mem Branch7Nonces;
  cl_mem Branch8Nonces;
  cl_mem Branch9Nonces;
  cl_mem Branch10Nonces;
  cl_mem Branch11Nonces;
  cl_mem Branch12Nonces;
  cl_mem Branch13Nonces;
  cl_mem Branch14Nonces;
  cl_mem Branch15Nonces;
  cl_mem Branch16Nonces;
  size_t GlobalThreadCount;
  unsigned char cldata[80];
  bool goffset;
  cl_uint vwidth;
  size_t max_work_size;
  size_t wsize;
  size_t compute_shaders;
} _clState;

extern int clDevicesNum(void);
extern _clState *initCl(unsigned int gpu, char *name, size_t nameSize, algorithm_t *algorithm);

#endif /* OCL_H */
