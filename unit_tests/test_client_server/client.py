"""最小化测试客户端 - 运行在 conda_env_a (Python 3.8)"""
import logging
import time

import numpy as np

from ipc_core import WebsocketClientPolicy

logging.basicConfig(level=logging.INFO, format='[CLIENT] %(message)s')

def capture_data(img_height=360, img_width=640, pcd_points=5000):
    """模拟采集图像和点云数据
    
    Args:
        img_height: 图像高度（默认 360，720p 的一半）
        img_width: 图像宽度（默认 640，720p 的一半）
        pcd_points: 点云点数（默认 5000，原来的一半）
    """
    # 模拟 RGB 图像（默认 360p，减少数据量）
    img = np.random.randint(0, 255, (img_height, img_width, 3), dtype=np.uint8)
    # 模拟点云（默认 5000 点，减少数据量）
    pcd = np.random.randn(pcd_points, 3).astype(np.float32)
    return {"image": img, "pcd": pcd}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_height", type=int, default=360, help="图像高度 (默认 360)")
    parser.add_argument("--img_width", type=int, default=640, help="图像宽度 (默认 640)")
    parser.add_argument("--pcd_points", type=int, default=5000, help="点云点数 (默认 5000)")
    args = parser.parse_args()
    
    # 计算数据大小
    img_size_mb = (args.img_height * args.img_width * 3) / 1024 / 1024
    pcd_size_mb = (args.pcd_points * 3 * 4) / 1024 / 1024
    total_size_mb = img_size_mb + pcd_size_mb
    print(f"数据大小: 图像 {img_size_mb:.2f} MB, 点云 {pcd_size_mb:.2f} MB, 总计 {total_size_mb:.2f} MB")
    
    # 建立连接
    client = WebsocketClientPolicy(host="127.0.0.1", port=8765)
    print("✅ 客户端已连接到策略服务器")

    # 循环调用
    try:
        step = 0
        latencies = []
        while True:
            start_t = time.time()
            
            # 采集数据
            data_start = time.time()
            obs = capture_data(args.img_height, args.img_width, args.pcd_points)
            data_time = (time.time() - data_start) * 1000
            
            # 发送并接收结果
            infer_start = time.time()
            action = client.infer(obs)
            infer_time = (time.time() - infer_start) * 1000
            
            # 计算总延迟
            total_latency = (time.time() - start_t) * 1000
            latencies.append(total_latency)
            
            if step % 10 == 0:
                avg_latency = sum(latencies[-10:]) / min(10, len(latencies))
                max_latency = max(latencies[-10:])
                min_latency = min(latencies[-10:])
                print(f"Step {step:04d} | 总延迟: {total_latency:.2f} ms | "
                      f"数据准备: {data_time:.2f} ms | 推理: {infer_time:.2f} ms | "
                      f"平均: {avg_latency:.2f} ms | 最大: {max_latency:.2f} ms | 最小: {min_latency:.2f} ms")
            
            step += 1
            
    except KeyboardInterrupt:
        if latencies:
            print(f"\n统计: 总次数={len(latencies)}, 平均延迟={sum(latencies)/len(latencies):.2f} ms, "
                  f"最大={max(latencies):.2f} ms, 最小={min(latencies):.2f} ms")
        print("用户中断")

if __name__ == "__main__":
    main()

