/*
Author: Tobias Fischer
*/

#include <math.h>
#include <stdio.h>


#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG


__device__ inline void world_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    // transform a point from world coordinate to local coordinate
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline void directions_to_local_coords(const float *dir, const float *box3d, float &local_dx, float &local_dy, float &local_dz){
    // transform a direction from world coordinate to local coordinate
    local_dz = dir[2];
    world_to_local_coords(dir[0], dir[1], box3d[6], local_dx, local_dy);
}


__device__ inline bool ray_aabb_intersect(float *ray_o, float *ray_d, float *aabb, float& tmin, float& tmax) {    
    float tmin_temp{};
    float tmax_temp{};

    float inv_dir[3] = {1.0f / ray_d[0], 1.0f / ray_d[1], 1.0f / ray_d[2]};

    if (inv_dir[0] >= 0) {
        tmin = (aabb[0] - ray_o[0]) * inv_dir[0];
        tmax = (aabb[3] - ray_o[0]) * inv_dir[0];
    } else {
        tmin = (aabb[3] - ray_o[0]) * inv_dir[0];
        tmax = (aabb[0] - ray_o[0]) * inv_dir[0];
    }

    if (inv_dir[1] >= 0) {
        tmin_temp = (aabb[1] - ray_o[1]) * inv_dir[1];
        tmax_temp = (aabb[4] - ray_o[1]) * inv_dir[1];
    } else {
        tmin_temp = (aabb[4] - ray_o[1]) * inv_dir[1];
        tmax_temp = (aabb[1] - ray_o[1]) * inv_dir[1];
    }

    if (tmin > tmax_temp || tmin_temp > tmax) return false;
    if (tmin_temp > tmin) tmin = tmin_temp;
    if (tmax_temp < tmax) tmax = tmax_temp;

    if (inv_dir[2] >= 0) {
        tmin_temp = (aabb[2] - ray_o[2]) * inv_dir[2];
        tmax_temp = (aabb[5] - ray_o[2]) * inv_dir[2];
    } else {
        tmin_temp = (aabb[5] - ray_o[2]) * inv_dir[2];
        tmax_temp = (aabb[2] - ray_o[2]) * inv_dir[2];
    }

    if (tmin > tmax_temp || tmin_temp > tmax) return false;
    if (tmin_temp > tmin) tmin = tmin_temp;
    if (tmax_temp < tmax) tmax = tmax_temp;

    if (tmax <= 0) return false;

    tmin = fmaxf(tmin, 0.0f);  // near plane
    tmax = fminf(tmax, INFINITY); // far plane
    return true;
}


__device__ inline int ray_box_intersect(const float *origin, const float *direction, const float *box3d, float *local_o, float *local_d, float *near_far){
    // param origin: (x, y, z)
    // param direcion: (dx, dy, dz)
    // param box3d: [x, y, z, dim x, dim y, dim z, heading] (x, y, z) is the box center

    // if box has zero dimension, return false
    if (box3d[3] <= 0.0 || box3d[4] <= 0.0 || box3d[5] <= 0.0){
        return false;
    }

    float x = origin[0], y = origin[1], z = origin[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    // transform to local
    local_o[2] = z - cz;
    world_to_local_coords(x - cx, y - cy, rz, local_o[0], local_o[1]);
    directions_to_local_coords(direction, box3d, local_d[0], local_d[1], local_d[2]);

    // ray-aabb intersect in local coords
    float local_dim[6] = {-dx/2, -dy/2, -dz/2, dx/2, dy/2, dz/2};
    return ray_aabb_intersect(local_o, local_d, local_dim, near_far[0], near_far[1]);
}


__global__ void assign_rays_to_box3d(int num_rays, int boxes_num, const float *origins, const float *dirs, const float *boxes3d, float *local_origins, float *local_directions, float *near_fars, bool *hit_mask){
    // params origins: (N, 3)
    // params dirs: (N, 3)
    // params boxes3d: (N, M, 7)
    // params local_origins: (N, M, 3)
    // params local_directions: (N, M, 3)
    // params near_fars: (N, M, 2)
    // params hit_mask: (N, M): true if intesects with the box
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int box_idx = blockIdx.y;

    if (ray_idx >= num_rays || box_idx >= boxes_num){
        return;
    }

    int box_offset = ray_idx * boxes_num * 7 + box_idx * 7;
    int ray_offset = ray_idx * 3;

    float local_o[3], local_d[3], near_far[2];
    bool hit = ray_box_intersect(origins + ray_offset, dirs  + ray_offset, boxes3d + box_offset, local_o, local_d, near_far);

    if (hit){
        // set local origin, direction, box index
        int n_m_idx = ray_idx * boxes_num + box_idx;
        local_origins[n_m_idx * 3] = local_o[0];
        local_origins[n_m_idx * 3 + 1] = local_o[1];
        local_origins[n_m_idx * 3 + 2] = local_o[2];

        hit_mask[n_m_idx] = true;

        local_directions[n_m_idx * 3] = local_d[0];
        local_directions[n_m_idx * 3 + 1] = local_d[1];
        local_directions[n_m_idx * 3 + 2] = local_d[2];

        near_fars[n_m_idx * 2] = near_far[0];
        near_fars[n_m_idx * 2 + 1] = near_far[1];
    }
}


void ray_box_intersect_Launcher(int num_rays, int boxes_num, const float *origins, const float *dirs, 
            const float *boxes3d, float *local_origins, float *local_directions, float *near_fars, bool *hit_mask){

    dim3 blocks(DIVUP(num_rays, THREADS_PER_BLOCK), boxes_num);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(THREADS_PER_BLOCK);
    assign_rays_to_box3d<<<blocks, threads>>>(num_rays, boxes_num, origins, dirs, boxes3d, local_origins, local_directions, near_fars, hit_mask);

#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}