#!/usr/bin/python3
# 文件名：client.py

# 导入 socket、sys 模块
import socket
import sys
import time

for i in range(3):
    # 创建 socket 对象
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取本地主机名
    host = socket.gethostname()

    # 设置端口号
    port = 8888

    # 连接服务，指定主机和端口

    s.connect((host, port))
    parameter = str(1.001+i)
    s.send(parameter.encode('utf-8'))
    print(i)

    # 接收小于 1024 字节的数据
    msg = s.recv(1024)
    print (msg.decode('utf-8'))

    time.sleep(5)  # 模拟实验，需要等待时间

    s.close()
