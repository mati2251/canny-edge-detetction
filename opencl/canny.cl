__kernel void canny(__global short * in, __global short *out)
{
  int x = get_global_id(0); 
  int y = get_global_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);
  int i = x + y * width;
  out[i] = in[i]/2;
}
