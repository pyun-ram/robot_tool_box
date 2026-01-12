import sys
sys.path.insert(0, '/home/astribot/Desktop/base_ws/robot_tool_box')

try:
    import openpi_client
    print("openpi_client imported successfully")
    from openpi_client import msgpack_numpy
except ImportError as e:
    print(f"Error importing openpi_client: {e}")