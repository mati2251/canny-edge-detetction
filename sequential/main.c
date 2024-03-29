#include <lodepng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct Image {
  short *data;
  unsigned short width;
  unsigned short height;
};

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

struct Image decode_image_gray(const char *filename) {
  unsigned error;
  unsigned width, height;
  unsigned char *data;
  error = lodepng_decode32_file(&data, &width, &height, filename);
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
  struct Image *image;
  image = malloc(width * height * sizeof(short) + 2 * sizeof(unsigned short));
  image->data = gray_scale_image(data, width, height);
  image->width = width;
  image->height = height;
  free(data);
  return *image;
}

void encode_image(struct Image *image, const char *filename) {
  unsigned error;

  char *new_filename = malloc(strlen(filename) + 6);
  strcpy(new_filename, filename);
  new_filename[strlen(new_filename) - 4] = '\0';
  strcat(new_filename, "_edge.png");

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
  if (error)
    printf("error %u: %s\n", error, lodepng_error_text(error));
}

short *conv(struct Image *image, float *kernel, short kernel_size) {
  short border = kernel_size / 2;
  short *out = malloc(image->width * image->height * sizeof(short));
  for (unsigned i = 0; i < image->width * image->height; i++) {
    out[i] = 0;
  }

  for (unsigned short i = 0; i < image->width; i++) {
    for (unsigned short j = 0; j < image->height; j++) {
      for (short m = -border; m <= border; m++) {
        for (short n = -border; n <= border; n++) {
          if ((short)(m + i) >= 0 && i + m < image->width &&
              (short)(j + n) >= 0 && j + n < image->height) {
            out[j * image->width + i] +=
                image->data[(j + n) * image->width + i + m] *
                kernel[(n + border) * kernel_size + (m + border)];
          }
        }
      }
    }
  }
  return out;
}

float *guassian_kernel(short size) {
  float *kernel = malloc(size * sizeof(float));
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
  return kernel;
}

void gaussian_filter(struct Image *image, short kernel_size) {
  float *kernel = guassian_kernel(kernel_size);
  short *filtered = conv(image, kernel, kernel_size);
  free(image->data);
  image->data = filtered;
}

short *sobel_x(struct Image *image) {
  float kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  return conv(image, kernel, 3);
}

short *sobel_y(struct Image *image) {
  float kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  return conv(image, kernel, 3);
}

void normalize(short *data, int size) {
  short max = 0, min = 0;
  for (unsigned short i = 0; i < size; i++) {
    max = max > data[i] ? max : data[i];
    min = min < data[i] ? min : data[i];
  }
  for (unsigned short i = 0; i < size; i++) {
    data[i] = (float)(data[i] - min) / (max - min) * 255;
  }
}

short *gradinet_direction(short *gradient_x, short *gradient_y,
                          unsigned short height, short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned short i = 0; i < height * width; i++) {
    out[i] = 0;
  }
  for (unsigned short i = 0; i < width; i++) {
    for (unsigned short j = 0; j < height; j++) {
      short gx = gradient_x[j * width + i];
      short gy = gradient_y[j * width + i];
      out[j * width + i] = atan2(gy, gx) * 180 / M_PI;
    }
  }
  return out;
}

short *gradient_intensity(short *gradient_x, short *gradient_y,
                          short int height, short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned short i = 0; i < height * width; i++) {
    out[i] = 0;
  }

  for (unsigned short i = 0; i < width; i++) {
    for (unsigned short j = 0; j < height; j++) {
      short gx = gradient_x[j * width + i];
      short gy = gradient_y[j * width + i];
      out[j * width + i] = sqrt(gx * gx + gy * gy);
    }
  }

  return out;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: %s <filename>\n", argv[0]);
    return 1;
  }
  const char *filename = argv[1];
  struct Image image = decode_image_gray(filename);

  printf("Image width: %d\n", image.width);
  printf("Image height: %d\n", image.height);

  gaussian_filter(&image, 5);

  short *gradient_y = sobel_y(&image);
  short *gradient_x = sobel_x(&image);
  short *gradient_int =
      gradient_intensity(gradient_x, gradient_y, image.height, image.width);
  normalize(gradient_int, image.height * image.width);
  short *gradient_dir =
      gradinet_direction(gradient_x, gradient_y, image.height, image.width);
  normalize(gradient_dir, image.height * image.width);

  image.data = gradient_dir;
  encode_image(&image, filename);
  free(image.data);
  return 0;
}
