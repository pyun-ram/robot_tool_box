from pathlib import Path
from tools.run import Runner
from multiprocessing import Pool
import os
from PIL import Image
import numpy as np

def process_episode(episode_path):
    """处理单个 episode 的包装函数"""
    episode = Path(episode_path)
    runner = Runner()
    # runner.convert_data_to_standard_format(
    #     episode_dir=str(episode),
    #     save_dir=f"data/rmt/20260114_s1_data_converted/train/pick_moving_target_from_belt/all_variations/episodes/{episode.name}",
    # )
    # max_frame_idx = len(list(Path(f"data/rmt/20260114_s1_data_converted/train/pick_moving_target_from_belt/all_variations/episodes/{episode.name}/front_rgb").glob("*.png"))) - 1
    # for idx in range(0, max_frame_idx, 10):
    #     runner.visualize_standard_format(
    #         data_dir=f"data/rmt/20260114_s1_data_converted/train/pick_moving_target_from_belt/all_variations/episodes/{episode.name}",
    #         frame_idx=idx,
    #     )
    # runner.visualize_calibration(
    #     save_dir=f"data/20260107_s1_data_vis/pick_moving_target_from_belt/{episode.name}",
    #     output_gif_path=f"data/20260107_s1_data_vis/pick_moving_target_from_belt/gif/{episode.name}.gif",
    # )
    return episode.name

if __name__ == "__main__":
    # get episode list (转换为列表以支持多进程)
    # episode_list = list(Path("data/20260114_s1_data/pick_moving_target_from_belt/").glob("episode*"))
    # num_processes = 20
    
    # print(f"找到 {len(episode_list)} 个 episode，使用 {num_processes} 个进程并行处理...")
    
    # # 使用多进程池处理
    # with Pool(processes=num_processes) as pool:
    #     results = pool.map(process_episode, episode_list)
    
    # print(f"完成！已处理 {len(results)} 个 episode")

    # runner = Runner()
    # runner.check_calibration_one_task(
    #     out_root=f"data/20260114_s1_data/pick_moving_target_from_belt/",
    #     save_base_dir=f"data/20260114_s1_data_vis/pick_moving_target_from_belt/",
    #     frame_skip=20,
    #     num_processes=10,
    # )
    # runner.check_calibration_one_task(
    #     out_root=f"data/20260107_s1_data/pick_moving_target_from_belt/",
    #     save_base_dir=f"data/20260107_s1_data_vis/pick_moving_target_from_belt/",
    #     frame_skip=20,
    #     num_processes=10,
    # )
    # runner.check_calibration_one_task(
    #     out_root=f"data/20260111_s1_data/pick_moving_target_from_belt/",
    #     save_base_dir=f"data/20260111_s1_data_vis/pick_moving_target_from_belt/",
    #     frame_skip=20,
    #     num_processes=10,
    # )
    vis_dir = 'data/20260114_s1_data_vis/pick_moving_target_from_belt'
    save_dir = 'data/20260114_s1_data_vis/pick_moving_target_from_belt_vis_idx0/'
    
    # create save_dir if not exists
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # get all episode directories
    vis_path = Path(vis_dir)
    episode_dirs = sorted([d for d in vis_path.iterdir() if d.is_dir() and d.name.startswith('episode')])
    
    print(f"找到 {len(episode_dirs)} 个 episode，开始处理...")
    
    for episode_dir in episode_dirs:
        episode_name = episode_dir.name
        # get four camera idx0 images
        cam_images = []
        for cam_id in range(1, 5):
            img_path = episode_dir / f'cam{cam_id}_idx0_projection.png'
            if img_path.exists():
                img = Image.open(img_path)
                cam_images.append(img)
            else:
                print(f"警告: {img_path} 不存在，跳过 {episode_name}")
                break
        
        # if we have all 4 images, concatenate them together
        if len(cam_images) == 4:
            # arrange in 2x2 grid
            # Calculate dimensions for the combined image
            img_width = cam_images[0].width
            img_height = cam_images[0].height
            
            # Create a new image with 2x2 grid layout
            combined_img = Image.new('RGB', (img_width * 2, img_height * 2))
            
            # Paste images in order: cam1 (top-left), cam2 (top-right), cam3 (bottom-left), cam4 (bottom-right)
            combined_img.paste(cam_images[0], (0, 0))  # cam1: top-left
            combined_img.paste(cam_images[1], (img_width, 0))  # cam2: top-right
            combined_img.paste(cam_images[2], (0, img_height))  # cam3: bottom-left
            combined_img.paste(cam_images[3], (img_width, img_height))  # cam4: bottom-right
            
            # save into save_dir with episode name
            save_path = Path(save_dir) / f'{episode_name}.png'
            combined_img.save(save_path)
            print(f"已保存: {save_path}")
        else:
            print(f"跳过 {episode_name}: 缺少图像文件")
    
    print(f"完成！所有图像已保存到 {save_dir}")
        
