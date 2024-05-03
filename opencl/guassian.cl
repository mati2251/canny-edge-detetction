#define SIZE 8 

__constant int guassian_kernel[3][3] = {
  {1, 2, 1},
  {2, 4, 2},
  {1, 2, 1}
};

__constant int float_sum = 16;


__kernel void guassian(__global short * in, __global short *out)
{
  int sum = 0;
  int x = get_global_id(0); 
  int y = get_global_id(1);
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);
  int i = x + y * width;  

  __local short local_data[2 + SIZE][2 + SIZE];
 
  local_data[local_x + 1][local_y + 1] = in[i];

  if (local_x == 0){
    local_data[local_x][local_y + 1] = in[i - 1];
    if (local_y == 0){
      local_data[local_x][local_y] = in[i - width - 1];
    }
    if (local_y == SIZE - 1){
      local_data[local_x][local_y + 2] = in[i + width - 1];
    }  
  }
  
  if (local_x == SIZE - 1){
    local_data[local_x + 2][local_y + 1] = in[i + 1];
    if (local_y == 0){
      local_data[local_x + 2][local_y] = in[i - width + 1];
    }
    if (local_y == SIZE - 1){
      local_data[local_x + 2][local_y + 2] = in[i + width + 1];
    }
  }
  
  if (local_y == 0){
    local_data[local_x + 1][local_y] = in[i - width];
  } 
  
  if (local_y == SIZE - 1){
    local_data[local_x + 1][local_y + 2] = in[i + width];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = 0; i < 3; i++){
    for (int j = 0; j < 3; j++){
      sum += (local_data[local_x][local_y] * guassian_kernel[i][j] / float_sum);
    }
  }

  out[i] = sum; 
  return;
}
