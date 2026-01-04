import asyncio
import websockets
import msgpack
from openpi_client import msgpack_numpy as m
import numpy as np

async def test_server(uri):
    try:
        # Connect to the server
        async with websockets.connect(uri) as websocket:
            print(f"Connected to server at {uri}")

            # Example observation to send
            observation = {
                "rgbs": np.zeros((4, 3, 256, 256)).tolist(),
                "pcds": np.zeros((4, 3, 256, 256)).tolist(),
                "curr_gripper": np.zeros((1, 8)).tolist(),
                "curr_gripper_history": np.zeros((3, 8)).tolist(),
                "instr": np.zeros((53, 512)).tolist(),
            }
            packed_obs = m.Packer().pack(observation)

            # Send the observation
            await websocket.send(packed_obs)
            print(f"Sent: {observation}")

            # Receive the response
            response = await websocket.recv()
            unpacked_response = m.unpackb(response)
            print(f"Received: {unpacked_response}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Replace with your server's URI
    server_uri = "ws://10.11.5.2:8000"
    asyncio.run(test_server(server_uri))