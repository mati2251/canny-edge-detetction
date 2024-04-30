#pragma once

struct Image {
    unsigned width;
    unsigned height;
    short *data;
};

short *gray_scale_image(unsigned char *data, unsigned width, unsigned height);

struct Image *decode_image_gray(const char *filename); 

void encode_image(struct Image *image, const char *filename,
                  const char *prefix); 
