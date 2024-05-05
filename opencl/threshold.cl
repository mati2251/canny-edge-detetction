#define SIZE 8

short one_threshold(short value, short high_threshold, short low_threshold)
{
  if (value >= high_threshold){
    return 255;
  } else if (value < low_threshold){
    return 0;
  } else {
    return 125;
  }
}

__kernel void threshold(__global short *in, __global short *output, short high_threshold, short low_threshold)
{
  int x = get_global_id(0); 
  int y = get_global_id(1);
  int local_x = get_local_id(0);
  int local_y = get_local_id(1);
  int width = get_global_size(0);
  int height = get_global_size(1);
  int i = x + y * width;

  __local short local_data[2 + SIZE][2 + SIZE];
 
  local_data[local_x + 1][local_y + 1] = one_threshold(in[i], high_threshold, low_threshold);
  if (local_x == 0){
    local_data[local_x][local_y + 1] = one_threshold(in[i - 1], high_threshold, low_threshold);
    if (local_y == 0){
      local_data[local_x][local_y] = one_threshold(in[i - width - 1], high_threshold, low_threshold);
    }
    if (local_y == SIZE - 1){
      local_data[local_x][local_y + 2] = one_threshold(in[i + width - 1], high_threshold, low_threshold);
    }  
  }
  
  if (local_x == SIZE - 1){
    local_data[local_x + 2][local_y + 1] = one_threshold(in[i + 1], high_threshold, low_threshold);
    if (local_y == 0){
      local_data[local_x + 2][local_y] = one_threshold(in[i - width + 1], high_threshold, low_threshold);
    }
    if (local_y == SIZE - 1){
      local_data[local_x + 2][local_y + 2] = one_threshold(in[i + width + 1], high_threshold, low_threshold);
    }
  }
  
  if (local_y == 0){
    local_data[local_x + 1][local_y] = one_threshold(in[i - width], high_threshold, low_threshold);
  } 
  
  if (local_y == SIZE - 1){
    local_data[local_x + 1][local_y + 2] = one_threshold(in[i + width], high_threshold, low_threshold);
  }
 
  barrier(CLK_LOCAL_MEM_FENCE);
  if (local_data[local_x + 1][local_y + 1] == 125){
    if (local_data[local_x][local_y + 1] == 255 || 
        local_data[local_x + 2][local_y + 1] == 255 || 
        local_data[local_x + 1][local_y] == 255 || 
        local_data[local_x + 1][local_y + 2] == 255 ||
        local_data[local_x][local_y] == 255 ||
        local_data[local_x + 2][local_y] == 255 ||
        local_data[local_x][local_y + 2] == 255 ||
        local_data[local_x + 2][local_y + 2] == 255
      ){
      local_data[local_x + 1][local_y + 1] = 255;
    } else {
      local_data[local_x + 1][local_y + 1] = 0;
    }
  }
  output[i] = local_data[local_x + 1][local_y + 1];
  return;
}
