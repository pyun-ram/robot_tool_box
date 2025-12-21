import fire

@dataclass
class Config:
    camera_names: ['cam0', 'cam1', 'cam2', 'cam3']
    camera_frequency: 20
    camera_resolution: (480, 270)
    camera_sync_precision: 8 # ms

class Runner:
    def organize_data(self, raw_episode_dir: str, episode_dir: str) -> None:
        # get start timestamp
        # get timestamp list
        # for each time stamp
        ## get closest cam0 data, and check its timestamp is within camera_sync_precision
        ## get closest cam1 data, and check its timestamp is within camera_sync_precision
        ## get closest cam2 data, and check its timestamp is within camera_sync_precision
        ## get closest cam3 data, and check its timestamp is within camera_sync_precision
        ## get closest gripper data, and check its timestamp is within camera_sync_precision
        ## save to save_episode_dir
        raise NotImplementedError("Not implemented")
    
    def check_calibration(self, episode_dir: str) -> None:
        # for each idx
        ## visualize gripper end-effector pose in cam0 pointcloud
        ## visualize gripper end-effector pose in cam1 pointcloud
        ## visualize gripper end-effector pose in cam2 pointcloud
        ## visualize gripper end-effector pose in cam3 pointcloud
        ## visualize merged pointcloud [cam0, cam1, cam2, cam3, gripper]
        raise NotImplementedError("Not implemented")

if __name__ == "__main__":
    fire.Fire(Runner)
