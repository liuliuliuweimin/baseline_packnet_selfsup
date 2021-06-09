# NuScenes



## Original

**Scenes:**  850 (train + val) => 200+ key frame



**Key Frame** 

* 0.5s sample 一次， lidar和image是同时刻
* 一个lidar，6个相机 
* Global 坐标系，ego pose， camera pose



其余的image 每隔0.1s 就存了一次



## Converted

对于一个sample我们需要什么：

对于不需要6个camera的：

prev, now, next

1. img: img_path 3个
2. cam_intrin: 3个
3. cam_pose:  3个； 相对于第三方坐标系
4. depth:  points_path.  1个



核心数据结构

train - val

1. 时间组织文件： prev， now， next索引文件: `/public/MARS/datasets/nuScenes-SF/meta/spatial_temp_train_v2.json`
   List{dict{'now': {filenames: [], cam_token: []}}}， 注意是6个相机,顺序是固定, now一定是key frame。 

2. Cam token 去索引 cam pose， cam intrin.  `/public/MARS/datasets/nuScenes-SF/meta/cam_pose_intrinsic_v3.json`, pose 是c 2 w的，这个w是在同一个scenes 里不变的坐标系

3. Depth-gt:  用image name做索引 `/public/MARS/datasets/nuScenes-SF/meta/two_cam_meta/spatial_temp_merged_path_train.json`

   