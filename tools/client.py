# run in the white pc
# 1. 启动一次 build connection to server

## 一直执行 rosspin？
## collect 4x real-sense camera data from ros node (480 x 270) 这块是接收 ros node 消息
## 接收后 sync into 20 Hz
## 同步后 send to server 远程调用 policy model

## 接收到 server 消息后，update action buffer （Time ensemble)

## 建立 Websocket 连接 和 远程调用 policy model 可以参考 tmp/openpi_client/websocket_client_policy.py