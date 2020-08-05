#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#include <fstream>
#include <iostream>
#define N 40
// #define BLOCK_SIZE 1
using namespace std;
char *loadProgSource(const char *filename, const char *preamble, size_t *sz)
{
  FILE *fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char *sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *)calloc(szSource + szPreamble + 1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}
void printMatrix(float *results)
{
  printf("output matrix: \n");
  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
      printf("%f\t", results[i * N + j]);
    }
    printf("\n");
  }
}
int main(int argc, char *argv[])
{
  // check input parameter
  if (argc != 2)
  {
    printf("Please input block size after program name, exiting!\n");
    exit(1);
  }
  int BLOCK_SIZE = atoi(argv[1]);

  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;
  cl_command_queue command_queue;
  char *kernelSource;
  size_t kernelSize;
  cl_program program;
  cl_kernel kernel;
  cl_mem inputA, inputB, output, total_size, Bwrk;
  size_t global[2], local[2];
  int i, j;
  cl_event prof_event;
  double run_time;

  // init two input matrix and one result matrix
  cl_float *inputMatrix1;
  cl_float *inputMatrix2;
  cl_float *results;
  cl_uint width = N;
  int x, y;
  int data = 0;
  inputMatrix1 = (cl_float *)malloc(sizeof(cl_float) * width * width);
  inputMatrix2 = (cl_float *)malloc(sizeof(cl_float) * width * width);
  results = (cl_float *)malloc(sizeof(cl_float) * width * width);
  for (y = 0; y < width; y++)
  {
    for (x = 0; x < width; x++)
    {
      inputMatrix1[y * width + x] = data;
      inputMatrix2[y * width + x] = data;
      results[y * width + x] = 0;
      data++;
      // printf("%f ", inputMatrix1[y * width + x]);
    }
    // printf("\n");
  }

  // Retrives a list of platforms available
  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS)
  {
    printf("Unable to get platform_id\n");
    return 1;
  }

  // Get a supported GPU device
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
                     &num_of_devices) != CL_SUCCESS)
  {
    printf("Unable to get device_id\n");
    return 1;
  }

  // Context properties list (must be terminated with 0)
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties)platform_id;
  properties[2] = 0;

  // Create a context with the GPU device
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

  // Create a command queue using the context and device
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

  // Load kernel file, prepend static info, and return total kernel size
  kernelSource = loadProgSource("matrix_multiplication.cl", "", &kernelSize);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS)
  {
    printf("Error building program\n");
    // Retrieve the latest compilation results
    char buffer[4096];
    size_t length;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    printf("compilation results = %s\n", buffer);
    exit(1);
  }

  // Specify which kernel from the program to execute
  kernel = clCreateKernel(program, "mat_mul", &err);

  // Create buffers for the input and output
  // total_size = clCreateBuffer(context, CL_MEM_READ_ONLY,
  //                             sizeof(int) * 1, NULL, NULL);
  inputA = clCreateBuffer(context, CL_MEM_READ_ONLY,
                          sizeof(cl_float) * width * width, NULL, NULL);
  inputB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                          sizeof(cl_float) * width * width, NULL, NULL);

  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                          sizeof(cl_float) * width * width, NULL, NULL);

  // Load data into the input buffer
  // int edge[1] = {N};
  // printf("edge =%d", edge[0]);
  // clEnqueueWriteBuffer(command_queue, total_size, CL_TRUE, 0,
  //                      sizeof(int) * 1, edge, 0, NULL, NULL);

  clEnqueueWriteBuffer(command_queue, inputA, CL_TRUE, 0,
                       sizeof(cl_float) * width * width, inputMatrix1, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, inputB, CL_TRUE, 0,
                       sizeof(cl_float) * width * width, inputMatrix2, 0, NULL, NULL);

  // Set the argument list for the kernel command
  int edge = N;
  clSetKernelArg(kernel, 0, sizeof(int), &edge);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &inputA);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &inputB);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &output);
  clSetKernelArg(kernel, 4, sizeof(cl_float) * width, NULL);
  global[0] = width;
  global[1] = width;

  local[0] = BLOCK_SIZE;
  local[1] = BLOCK_SIZE;

  // Enqueue the kernel command for execution
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
                         local, 0, NULL, &prof_event);
  //  synchronization point
  clFinish(command_queue);
  // Blocks until events are complete
  err = clWaitForEvents(1, &prof_event);
  // profiling
  cl_ulong start_time, end_time;
  size_t return_bytes;
  err = clGetEventProfilingInfo(
      prof_event,
      CL_PROFILING_COMMAND_START,
      sizeof(cl_ulong),
      &start_time,
      &return_bytes);

  err = clGetEventProfilingInfo(
      prof_event,
      CL_PROFILING_COMMAND_END,
      sizeof(cl_ulong),
      &end_time,
      &return_bytes);

  run_time = (double)(end_time -
                      start_time);
  printf("kernel execution time with block size %d in nanoseconds= %lf\n", BLOCK_SIZE, run_time);
  printf("kernel execution time with block size %d in milliseconds= %lf\n", BLOCK_SIZE, run_time / 1000000);
  // Copy the results from out of the output buffer
  clEnqueueReadBuffer(command_queue, output, CL_TRUE, 0,
                      sizeof(cl_float) * width * width, results, 0, NULL, NULL);
  // write result to file
  ofstream file;
  file.open("assessment3_result.txt");
  if (!file)
  {
    printf("cannot open the file");
    exit(1);
  }
  else
  {
    file << std::fixed;
    for (i = 0; i < width; i++)
    {
      for (j = 0; j < width; j++)
      {
        file << results[i * width + j] << " ";
      }
      file << "\n";
    }
    file.close();
  }

  // Print the results
  printMatrix(results);

  // Cleanup (release OpenCL resources)
  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(inputA);
  clReleaseMemObject(inputB);
  clReleaseMemObject(output);

  return 0;
}
