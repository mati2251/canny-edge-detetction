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
          if (m + (short)i >= 0 && i + m < image->width &&
              j + (short)n >= 0 && j + n < image->height) {
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
          exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
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

  short high_threshold = max * high_ratio;
  short low_threshold = max * low_ratio;
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
  struct Image *image = decode_image_gray(filename);

  printf("Image width: %d\n", image->width);
  printf("Image height: %d\n", image->height);
  encode_image(image, filename, "_gray");

  gaussian_filter(image, kernel_size, sigma);
  encode_image(image, filename, "_gaussian");
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

  free(image->data);

  normalize(gradient_x, image->width * image->height);
  image->data = gradient_x;
  encode_image(image, filename, "_sobel_x");

  normalize(gradient_y, image->width * image->height);
  image->data = gradient_y;
  encode_image(image, filename, "_sobel_y");

  image->data = gradient_int;
  encode_image(image, filename, "_gradient");

  normalize(gradient_dir, image->width * image->height);
  image->data = gradient_dir;
  encode_image(image, filename, "_direction");

  image->data = non_max;
  encode_image(image, filename, "_non_max");

  image->data = thre;
  encode_image(image, filename, "_threshold");

  image->data = hyste;
  encode_image(image, filename, "_edge");

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
