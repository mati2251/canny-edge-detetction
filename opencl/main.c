#include "../universal/universal.h"
#include <CL/cl.h>
#include <lodepng.h>
#include <omp.h>
#include <stdio.h>

#define ERR_HANDLER(CODE) err_handler(__FILE__, __LINE__, CODE)

void err_handler(const char *file, int line, cl_int code) {
  if (code != CL_SUCCESS) {
    fprintf(stderr, "Opencl error: %s %d: %d", file, line, code);
    exit(1);
  }
}

void get_kernel_err(cl_program program, cl_device_id device) {
  size_t log_size;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                        &log_size);

  char *log = (char *)malloc(log_size);

  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log,
                        NULL);
  printf("%s\n", log);
}

char *read_kernel(char *filename) {
  FILE *vadd_kernel_file = fopen(filename, "r");
  if (!vadd_kernel_file) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  fseek(vadd_kernel_file, 0L, SEEK_END);
  size_t vadd_kernel_size = ftell(vadd_kernel_file);
  fseek(vadd_kernel_file, 0L, SEEK_SET);
  char *vadd_kernel_source = (char *)malloc(vadd_kernel_size + 1);
  vadd_kernel_size =
      fread(vadd_kernel_source, 1, vadd_kernel_size, vadd_kernel_file);
  vadd_kernel_source[vadd_kernel_size] = '\0';
  fclose(vadd_kernel_file);
  return vadd_kernel_source;
}

void normalize(short *data, unsigned int size) {
  short max = 0, min = 0;
  for (unsigned int i = 0; i < size; i++) {
    max = max > data[i] ? max : data[i];
    min = min < data[i] ? min : data[i];
  }
  for (unsigned int i = 0; i < size; i++) {
    data[i] = (float)(data[i] - min) / (max - min) * 255;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    return 1;
  }
  const char *filename = argv[1];

  cl_platform_id platform_id = NULL;
  cl_device_id device = NULL;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ERR_HANDLER(ret);
  ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device,
                       &ret_num_devices);
  ERR_HANDLER(ret);

  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &ret);
  ERR_HANDLER(ret);
  cl_command_queue command_queue =
      clCreateCommandQueue(context, device, 0, &ret);
  ERR_HANDLER(ret);

  char *guassian_source = read_kernel("guassian.cl");

  cl_program guassian_program = clCreateProgramWithSource(
      context, 1, (const char **)&guassian_source, NULL, &ret);
  ERR_HANDLER(ret);

  ret = clBuildProgram(guassian_program, 1, &device, NULL, NULL, NULL);
  get_kernel_err(guassian_program, device);
  ERR_HANDLER(ret);

  cl_kernel guassian_kernel =
      clCreateKernel(guassian_program, "guassian", &ret);
  ERR_HANDLER(ret);

  char *sobel_source = read_kernel("sobel.cl");

  cl_program sobel_program = clCreateProgramWithSource(
      context, 1, (const char **)&sobel_source, NULL, &ret);
  ERR_HANDLER(ret);

  ret = clBuildProgram(sobel_program, 1, &device, NULL, NULL, NULL);
  get_kernel_err(sobel_program, device);
  ERR_HANDLER(ret);

  cl_kernel sobel_kernel = clCreateKernel(sobel_program, "sobel", &ret);
  ERR_HANDLER(ret);

  double start = omp_get_wtime();

  struct Image *image = decode_image_gray(filename);

  cl_mem input =
      clCreateBuffer(context, CL_MEM_READ_ONLY,
                     image->width * image->height * sizeof(short), NULL, &ret);
  ERR_HANDLER(ret);
  cl_mem guassian =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     image->width * image->height * sizeof(short), NULL, &ret);
  ERR_HANDLER(ret);

  cl_mem sobel_gradient =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     image->width * image->height * sizeof(short), NULL, &ret);
  ERR_HANDLER(ret);

  cl_mem sobel_direction =
      clCreateBuffer(context, CL_MEM_READ_WRITE,
                     image->width * image->height * sizeof(short), NULL, &ret);

  ret = clEnqueueWriteBuffer(command_queue, input, CL_TRUE, 0,
                             image->width * image->height * sizeof(short),
                             image->data, 0, NULL, NULL);
  ERR_HANDLER(ret);

  ret = clSetKernelArg(guassian_kernel, 0, sizeof(cl_mem), &input);
  ERR_HANDLER(ret);

  ret = clSetKernelArg(guassian_kernel, 1, sizeof(cl_mem), &guassian);
  ERR_HANDLER(ret);

  size_t global_work_size[2] = {image->width, image->height};
  size_t local_work_size[2] = {8, 8};
  ret =
      clEnqueueNDRangeKernel(command_queue, guassian_kernel, 2, NULL,
                             global_work_size, local_work_size, 0, NULL, NULL);
  ERR_HANDLER(ret);

  ret = clSetKernelArg(sobel_kernel, 0, sizeof(cl_mem), &guassian);
  ERR_HANDLER(ret);

  ret = clSetKernelArg(sobel_kernel, 1, sizeof(cl_mem), &sobel_gradient);
  ERR_HANDLER(ret);

  ret = clSetKernelArg(sobel_kernel, 2, sizeof(cl_mem), &sobel_direction);
  ERR_HANDLER(ret);

  ret =
      clEnqueueNDRangeKernel(command_queue, sobel_kernel, 2, NULL,
                             global_work_size, local_work_size, 0, NULL, NULL);
  ERR_HANDLER(ret);

  ret = clEnqueueReadBuffer(command_queue, sobel_gradient, CL_TRUE, 0,
                            image->width * image->height * sizeof(short),
                            image->data, 0, NULL, NULL);
  ERR_HANDLER(ret);
  
  double end = omp_get_wtime();
  
  clReleaseMemObject(guassian);
  clReleaseMemObject(sobel_gradient);
  clReleaseMemObject(input);
  clReleaseKernel(guassian_kernel);
  clReleaseKernel(sobel_kernel);
  clReleaseProgram(guassian_program);
  clReleaseProgram(sobel_program);
  clReleaseCommandQueue(command_queue);
  clReleaseContext(context);
  free(guassian_source);
  free(sobel_source);

  printf("Time: %f\n", end - start);
  encode_image(image, filename, "_opencl");
  return 0;
}
