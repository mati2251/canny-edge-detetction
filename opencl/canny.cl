#define SIZE 8 

__kernel void canny(__global short * intensity, __global short *direction, __global short *out)
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
 
  local_data[local_x + 1][local_y + 1] = intensity[i];
  if (local_x == 0){
    local_data[local_x][local_y + 1] = intensity[i - 1];
    if (local_y == 0){
      local_data[local_x][local_y] = intensity[i - width - 1];
    }
    if (local_y == SIZE - 1){
      local_data[local_x][local_y + 2] = intensity[i + width - 1];
    }  
  }
  
  if (local_x == SIZE - 1){
    local_data[local_x + 2][local_y + 1] = intensity[i + 1];
    if (local_y == 0){
      local_data[local_x + 2][local_y] = intensity[i - width + 1];
    }
    if (local_y == SIZE - 1){
      local_data[local_x + 2][local_y + 2] = intensity[i + width + 1];
    }
  }
  
  if (local_y == 0){
    local_data[local_x + 1][local_y] = intensity[i - width];
  } 
  
  if (local_y == SIZE - 1){
    local_data[local_x + 1][local_y + 2] = intensity[i + width];
  }
 
  barrier(CLK_LOCAL_MEM_FENCE);

  short dir = direction[i];
  short value = intensity[i];
  short r = 255, q = 255;
  if (dir < 0){
    dir += 180;
  }
  if (dir <= 22 || dir > 157) {

  } 
  if (dir <= 22 || dir > 157) {
    r = local_data[local_x + 1][local_y];
    q = local_data[local_x + 1][local_y+2]; 
  } else if (dir <= 67) {
    r = local_data[local_x][local_y+2];
    q = local_data[local_x+2][local_y]; 
  } else if (dir <= 112) {
    r = local_data[local_x][local_y + 1];
    q = local_data[local_x + 2][local_y + 1];
  } else if (dir <= 157) {
    r = local_data[local_x + 2][local_y + 2];
    q = local_data[local_x][local_y];
  }
  if (value >= r && value >= q) {
    out[i] = value;
  } else {
    out[i] = 0;
  }
  return;
}
