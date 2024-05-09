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
