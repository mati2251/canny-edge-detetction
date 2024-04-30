#include <omp.h>
#include <stdio.h>
#include <CL/cl.h>
#include <lodepng.h>
#include "../universal/universal.h"

int main(int argc, char **argv) {
  short kernel_size = 7;
  float low_ratio = 0.05;
  float high_ratio = 0.5;
  float sigma = 3;
  if (argc < 6 && argc != 2) {
    printf(
        "Usage: %s <filename> <sigma> <kernel_size> <high_ratio> <low_ratio>\n",
        argv[0]);
    return 1;
  }
  if (argc == 6) {
    sigma = atof(argv[2]);
    kernel_size = atoi(argv[3]);
    low_ratio = atof(argv[5]);
    high_ratio = atof(argv[4]);
  }
  const char *filename = argv[1];

  double start = omp_get_wtime();
  struct Image *image = decode_image_gray(filename);
  double end = omp_get_wtime();
  printf("Image loaded in %f seconds\n", end - start);
  encode_image(image, filename, "_opencl");
  return 0;
}
