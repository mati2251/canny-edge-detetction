#include <lodepng.h>
#include <stdio.h>
#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

void decode_image(const char* filename, unsigned char *image, unsigned *width, unsigned *height) {
  unsigned error;

  error = lodepng_decode32_file(&image, width, height, filename);
  if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    return 1;
  }
  const char* filename = "test.png";
  unsigned char *image = 0;
  unsigned width, height;
  decode_image(filename, image, &width, &height);
  printf("Image width: %d\n", width);
  printf("Image height: %d\n", height);
  free(image);
  return 0;
}

