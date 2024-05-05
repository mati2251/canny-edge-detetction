#include "../universal/universal.h"
#include <lodepng.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

short *conv(struct Image *image, float *kernel, short kernel_size) {
  short border = kernel_size / 2;
  short *out = malloc(image->width * image->height * sizeof(short));
  for (unsigned int i = 0; i < image->width * image->height; i++) {
    out[i] = 0;
  }

  for (unsigned short i = 0; i < image->width; i++) {
    for (unsigned short j = 0; j < image->height; j++) {
      for (short m = -border; m <= border; m++) {
        for (short n = -border; n <= border; n++) {
          if (m + (short)i >= 0 && i + m < (int)image->width &&
              j + (short)n >= 0 && j + n < (int)image->height) {
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

float *guassian_kernel(short size, float sigma) {
  float *kernel = malloc(size * size * sizeof(float));
  short border = size / 2;
  float sum = 0;
  for (short i = -border; i <= border; i++) {
    for (short j = -border; j <= border; j++) {
      kernel[(i + border) * size + (j + border)] =
          exp(-(i * i + j * j) / (2 * sigma * sigma)) /
          (2 * M_PI * sigma * sigma);
      sum += kernel[(i + border) * size + (j + border)];
    }
  }
  for (short i = 0; i < size; i++) {
    for (short j = 0; j < size; j++) {
      kernel[i * size + j] /= sum;
    }
  }
  return kernel;
}

void gaussian_filter(struct Image *image, short kernel_size, float sigma) {
  float *kernel = guassian_kernel(kernel_size, sigma);
  short *filtered = conv(image, kernel, kernel_size);
  free(image->data);
  image->data = filtered;
  free(kernel);
}

short *sobel_x(struct Image *image) {
  float kernel[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  return conv(image, kernel, 3);
}

short *sobel_y(struct Image *image) {
  float kernel[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  return conv(image, kernel, 3);
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

short *gradinet_direction(short *gradient_x, short *gradient_y,
                          unsigned short height, short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned short i = 0; i < width; i++) {
    for (unsigned short j = 0; j < height; j++) {
      short gx = gradient_x[j * width + i];
      short gy = gradient_y[j * width + i];
      out[j * width + i] = atan2(gy, gx) * 180 / M_PI;
    }
  }
  normalize(out, height * width);
  return out;
}

short *gradient_intensity(short *gradient_x, short *gradient_y,
                          short int height, short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned short i = 0; i < width; i++) {
    for (unsigned short j = 0; j < height; j++) {
      short gx = gradient_x[j * width + i];
      short gy = gradient_y[j * width + i];
      out[j * width + i] = sqrt(gx * gx + gy * gy);
    }
  }

  return out;
}

short *non_maximum(short *gradient_int, short *gradient_dir, short int height,
                   short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned short i = 0; i < width; i++) {
    out[i] = 0;
    out[(height - 1) * width + i] = 0;
  }
  for (unsigned short i = 0; i < height; i++) {
    out[i * width] = 0;
    out[i * width + width - 1] = 0;
  }

  for (unsigned short i = 1; i < width - 1; i++) {
    for (unsigned short j = 1; j < height - 1; j++) {
      short dir = gradient_dir[j * width + i];
      short value = gradient_int[j * width + i];
      short r = 255, q = 255;
      if (dir < 0) {
        dir += 180;
      }
      if (dir <= 22 || dir > 157) {
        r = gradient_int[(j - 1) * width + i];
        q = gradient_int[(j + 1) * width + i];
      } else if (dir <= 67) {
        r = gradient_int[(j + 1) * width + i - 1];
        q = gradient_int[(j - 1) * width + i + 1];
      } else if (dir <= 112) {
        r = gradient_int[j * width + i - 1];
        q = gradient_int[j * width + i + 1];
      } else if (dir <= 157) {
        r = gradient_int[(j + 1) * width + i + 1];
        q = gradient_int[(j - 1) * width + i - 1];
      }
      if (value >= r && value >= q) {
        out[j * width + i] = value;
      } else {
        out[j * width + i] = 0;
      }
    }
  }
  return out;
}

short *threshold(short *data, short int height, short int width,
                 float low_ratio, float high_ratio) {
  short *out = malloc(height * width * sizeof(short));

  short max = 0;
  for (unsigned int i = 0; i < (unsigned int)(height * width); i++) {
    max = max > data[i] ? max : data[i];
  }

  short high_threshold = max * 0.1;
  short low_threshold = high_threshold * 0.2;
  for (unsigned short i = 0; i < width; i++) {
    for (unsigned short j = 0; j < height; j++) {
      if (data[j * width + i] >= high_threshold) {
        out[j * width + i] = 255;
      } else if (data[j * width + i] < low_threshold) {
        out[j * width + i] = 0;
      } else {
        out[j * width + i] = 125;
      }
    }
  }

  return out;
}

short *hysterisis(short *data, short int height, short int width) {
  short *out = malloc(height * width * sizeof(short));
  for (unsigned int i = 0; i < (unsigned int)(height * width); i++) {
    out[i] = data[i];
  }

  for (unsigned short i = 1; i < width - 1; i++) {
    for (unsigned short j = 1; j < height - 1; j++) {
      if (data[j * width + i] == 125) {
        if (data[(j - 1) * width + i - 1] == 255 ||
            data[(j - 1) * width + i] == 255 ||
            data[(j - 1) * width + i + 1] == 255 ||
            data[j * width + i - 1] == 255 || data[j * width + i + 1] == 255 ||
            data[(j + 1) * width + i - 1] == 255 ||
            data[(j + 1) * width + i] == 255 ||
            data[(j + 1) * width + i + 1] == 255) {
          out[j * width + i] = 255;
        } else {
          out[j * width + i] = 0;
        }
      }
    }
  }
  return out;
}

int main(int argc, char *argv[]) {
  short kernel_size = 3;
  float low_ratio = 0.2;
  float high_ratio = 0.1;
  float sigma = 1;
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
  gaussian_filter(image, kernel_size, sigma);
  short *gradient_y = sobel_y(image);
  short *gradient_x = sobel_x(image);
  short *gradient_int =
      gradient_intensity(gradient_x, gradient_y, image->height, image->width);
  short *gradient_dir =
      gradinet_direction(gradient_x, gradient_y, image->height, image->width);
  short *non_max =
      non_maximum(gradient_int, gradient_dir, image->height, image->width);
  short *thre =
      threshold(non_max, image->height, image->width, low_ratio, high_ratio);
  short *hyste = hysterisis(thre, image->height, image->width);
  double end = omp_get_wtime();
  printf("Image width: %d\n", image->width);
  printf("Image height: %d\n", image->height);
  printf("Time: %f\n", end - start);

  // encode_image(image, filename, "_guassian");
  // free(image->data);
  //
  // normalize(gradient_x, image->width * image->height);
  // image->data = gradient_x;
  // encode_image(image, filename, "_sobel_x");
  //
  // normalize(gradient_y, image->width * image->height);
  // image->data = gradient_y;
  // encode_image(image, filename, "_sobel_y");
  //
  // image->data = gradient_int;
  // encode_image(image, filename, "_gradient");
  //
  // normalize(gradient_dir, image->width * image->height);
  // image->data = gradient_dir;
  // encode_image(image, filename, "_direction");
  //
  // image->data = non_max;
  // encode_image(image, filename, "_non_max");
  //
  // image->data = thre;
  // encode_image(image, filename, "_threshold");

  image->data = hyste;
  encode_image(image, filename, "_sequential");

  free(gradient_x);
  free(gradient_y);
  free(gradient_int);
  free(gradient_dir);
  free(non_max);
  free(thre);
  free(hyste);
  free(image);
  return 0;
}
