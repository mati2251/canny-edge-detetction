#include "universal.h"
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

short *gray_scale_image(unsigned char *data, unsigned width, unsigned height) {
  short *image = malloc(width * height * sizeof(short) * 4);
  for (unsigned i = 0; i < width; i++) {
    for (unsigned j = 0; j < height; j++) {
      unsigned char r = data[j * width * 4 + i * 4];
      unsigned char g = data[j * width * 4 + i * 4 + 1];
      unsigned char b = data[j * width * 4 + i * 4 + 2];
      image[j * width + i] = 0.3 * r + 0.59 * g + 0.11 * b;
    }
  }
  return image;
}

struct Image *decode_image_gray(const char *filename) {
  unsigned error;
  unsigned width, height;
  unsigned char *data;
  error = lodepng_decode32_file(&data, &width, &height, filename);
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
  struct Image *image;
  image = malloc(sizeof(struct Image));
  image->data = gray_scale_image(data, width, height);
  image->width = width;
  image->height = height;
  free(data);
  return image;
}

void encode_image(struct Image *image, const char *filename,
                  const char *prefix) {
  unsigned error;

  char *new_filename = malloc(strlen(filename) + strlen(prefix) + 4);
  strcpy(new_filename, filename);
  new_filename[strlen(new_filename) - 4] = '\0';
  strcat(new_filename, prefix);
  strcat(new_filename, ".png");

  unsigned char *data = malloc(image->width * image->height * 4);

  for (unsigned i = 0; i < image->width; i++) {
    for (unsigned j = 0; j < image->height; j++) {
      unsigned char value = image->data[j * image->width + i];
      data[j * image->width * 4 + i * 4] = value;
      data[j * image->width * 4 + i * 4 + 1] = value;
      data[j * image->width * 4 + i * 4 + 2] = value;
      data[j * image->width * 4 + i * 4 + 3] = 255;
    }
  }

  error =
      lodepng_encode32_file(new_filename, data, image->width, image->height);
  free(data);
  free(new_filename);
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
}
