# Canny edge detection
Canny edge detectiom is popular edge detection algorithm. This project is implementation that algorithm in 3 ways to compare the performance of each implementation. The 3 ways are:
* Sequential
* Parallel using OpenMP
* Parallel using OpenCL
## Algorithm steps
0. Orginal image
![Original image](images/1920x1080/1920x1080.png)
1. Convert the image to grayscale
![Grayscale image](images/1920x1080/1920x1080_gray.png)
2. Apply Gaussian filter to smooth the image in order to remove the noise
![Gaussian filter](images/1920x1080/1920x1080_gaussian.png)
3. Sobel filter
Sobel x
![Sobel filter](images/1920x1080/1920x1080_sobel_y.png)
Sobel y
![Sobel filter](images/1920x1080/1920x1080_sobel_x.png)
Sobel intensity
![Sobel intensity](images/1920x1080/1920x1080_gradient.png)
Sobel direction
![Sobel direction](images/1920x1080/1920x1080_direction.png)
4. Non-maximum suppression
![Non-maximum suppression](images/1920x1080/1920x1080_non_max.png)
5. Hysteresis thresholding
![Hysteresis thresholding](images/1920x1080/1920x1080_edge.png)
## Algorithm comprarison
### Tested device
* CPU: AMD Ryzen 5 2500U
* GPU: Radeon Vega 8 Gfx
## Performance comparison
Images used for creating algorithm (from images folder):
| Image Size | Sequential | OpenMP | OpenCL |
|------------|------------|--------|--------|
| 512x512    | 0.14s      | 0.07s  | 0.05s  |
| 512x512    | 0.13s      | 0.06s  | 0.04s  |
| 800x600    | 0.19s      | 0.11s  | 0.06s  |
| 1920x1080  | 0.90s      | 0.46s  | 0.29s  |
| 4032x2264  | 4.18s      | 1.99s  | 1.40s  |

Performence testing on diffrent size the same [image](https://www.flickr.com/photos/gsfc/6760135001/sizes/s/in/photostream/), only canny time (without image loading and saving):
| Image Size | Sequential | OpenMP | OpenCL | OpenCV |
|------------|------------|--------|--------|--------|
| 240x240    | 0.02s      | 0.00s  | 0.00s  | 0.46s  |
| 320x320    | 0.04s      | 0.01s  | 0.00s  | 0.47s  |
| 400x400    | 0.06s      | 0.02s  | 0.00s  | 0.46s  |
| 640x640    | 0.17s      | 0.05s  | 0.01s  | 0.46s  |
| 800x800    | 0.27s      | 0.08s  | 0.02s  | 0.46s  |
| 1024x1024  | 0.58s      | 0.11s  | 0.02s  | 0.47s  |
| 1600x1600  | 1.09s      | 0.25s  | 0.04s  | 0.47s  |
| 2048x2048  | 3.49s      | 0.52s  | 0.05s  | 0.50s  |
| 3072x3072  | 7.41s      | 1.13s  | 0.11s  | 0.54s  |
| 4096x4096  | 13.50s     | 2.79s  | 0.24s  | 0.60s  |
| 5120x5120  | 35.28s     | 4.06s  | 0.35s  | 0.60s  |
| 6144x6144  | 47.03s     | 12.55s | 0.51s  | 0.74s  |
