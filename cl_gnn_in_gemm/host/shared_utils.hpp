#include "CL/cl.hpp"
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#include <vector>

static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue1 = NULL;
static cl_command_queue queue2 = NULL;
map<string, cl_kernel> kernels;
static cl_program program = NULL;
int BLOCK_SIZE = 32;
int WPT = 2;
