#include "lodepng.h"
#include <lodepng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct Image {
  unsigned char *data;
  unsigned width;
  unsigned height;
};

void decode_image(struct Image *image, const char *filename) {
  unsigned error;
  error = lodepng_decode32_file(&image->data, &image->width, &image->height,
                                filename);
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
}

void encode_image(struct Image *image, const char *filename) {
  unsigned error;

  char *new_filename = malloc(strlen(filename) + 6);
  strcpy(new_filename, filename);
  new_filename[strlen(new_filename) - 4] = '\0';
  strcat(new_filename, "_edge.png");

  error = lodepng_encode32_file(new_filename, image->data, image->width,
                                image->height);
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
}

void gray_scale_image(struct Image *image){
  for (unsigned i = 0; i < image->width; i++) {
    for (unsigned j = 0; j < image->height; j++) {
      unsigned char r = image->data[j * image->width * 4 + i * 4];
      unsigned char g = image->data[j * image->width * 4 + i * 4 + 1];
      unsigned char b = image->data[j * image->width * 4 + i * 4 + 2];
      unsigned char gray = 0.3 * r + 0.59 * g + 0.11 * b;
      image->data[j * image->width * 4 + i * 4] = gray;
      image->data[j * image->width * 4 + i * 4 + 1] = gray;
      image->data[j * image->width * 4 + i * 4 + 2] = gray;
    }
  }
}

void guassian_kernel(int size, float *kernel) {
  float sigma = 1.0;
  float sum = 0.0;
  for (int i = 0; i < size; i++) {
    int x = i - size / 2;
    kernel[i] = exp(-x * x / (2 * sigma * sigma)) / (sqrt(2 * M_PI) * sigma);
    sum += kernel[i];
  }
  for (int i = 0; i < size; i++) {
    kernel[i] /= sum;
  }
}

void gaussian_filter(struct Image *image, int kernel_size) {
  int border = kernel_size / 2;
  float *kernel = malloc(kernel_size * sizeof(float));
  guassian_kernel(kernel_size, kernel);

  unsigned char *new_image = malloc(image->width * image->height * 4);
  for (unsigned i = 0; i < image->width * image->height * 4; i++) {
    new_image[i] = 0;
  }

  for (unsigned i = 0; i < image->width; i++) {
    for (unsigned j = 0; j < image->height; j++) {
      for (int m = -border; m <= border; m++) {
        for (int n = -border; n <= border; n++) {
          if ((int)(m + i) >= 0 && i + m < image->width && (int)(j + n) >= 0 &&
              j + n < image->height) {
            new_image[j * image->width * 4 + i * 4] +=
                image->data[(j + n) * image->width * 4 + (i + m) * 4] *
                kernel[m + border] * kernel[n + border];
            new_image[j * image->width * 4 + i * 4 + 1] +=
                image->data[(j + n) * image->width * 4 + (i + m) * 4 + 1] *
                kernel[m + border] * kernel[n + border];
            new_image[j * image->width * 4 + i * 4 + 2] +=
                image->data[(j + n) * image->width * 4 + (i + m) * 4 + 2] *
                kernel[m + border] * kernel[n + border];
            new_image[j * image->width * 4 + i * 4 + 3] = 255;
          }
        }
      }
    }
  }

  free(image->data);
  free(kernel);
  image->data = new_image;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    return 1;
  }
  const char *filename = argv[1];
  struct Image image;
  decode_image(&image, filename);

  printf("Image width: %d\n", image.width);
  printf("Image height: %d\n", image.height);
  
  gray_scale_image(&image);
  gaussian_filter(&image, 5);

  encode_image(&image, filename);
  free(image.data);
  return 0;
}
