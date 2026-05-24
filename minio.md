```bash
39.105.65.198
```
```
docker run -d --name minio \
-p 9000:9000 -p 9001:9001 \
-e "MINIO_ROOT_USER=minioadmin" \
-e "MINIO_ROOT_PASSWORD=minioadmin" \
-v $(pwd)/volumes/minio/data:/data \
quay.io/minio/minio server /data --console-address ":9001"

docker stop minio
docker start minio
docker logs -f minio
```

# 完整版
## 下载
```bash
wget https://dl.min.io/server/minio/release/linux-amd64/archive/minio-20230809233022.0.0.x86_64.rpm
```

## 安装MinIO
```bash
rpm -ivh minio-20230809233022.0.0.x86_64.rpm
```

## 集成Systemd
```bash
vim /etc/systemd/system/minio.service
```
### 编写MinIO服务配置文件
```text
[Unit]
Description=MinIO
Documentation=https://min.io/docs/minio/linux/index.html
Wants=network-online.target
After=network-online.target
AssertFileIsExecutable=/usr/local/bin/minio

[Service]
WorkingDirectory=/usr/local
ProtectProc=invisible
EnvironmentFile=-/etc/default/minio
ExecStartPre=/bin/bash -c "if [ -z \"${MINIO_VOLUMES}\" ]; then echo \"Variable MINIO_VOLUMES not set in /etc/default/minio\"; exit 1; fi"
ExecStart=/usr/local/bin/minio server $MINIO_OPTS $MINIO_VOLUMES
Restart=always
LimitNOFILE=65536
TasksMax=infinity
TimeoutStopSec=infinity
SendSIGKILL=no

[Install]
WantedBy=multi-user.target
```
### 编写EnvironmentFile文件

执行以下命令创建并打开`/etc/default/minio`文件
```bash
vim /etc/default/minio
```
内容如下，具体可参考官方文档。
```text
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_VOLUMES=/data
MINIO_OPTS="--console-address :9001"
```

#### 注意

- `MINIO_ROOT_USER`和`MINIO_ROOT_PASSWORD`为用于访问 MinIO 的用户名和密码，密码长度至少8位。
- `MINIO_VOLUMES`用于指定数据存储路径，需确保指定的路径是存在的，可执行以下命令创建该路径。
```bash
mkdir /data
chmod -R 777 /data
MINIO_OPTS中的console-address,用于指定管理页面的地址。
```
## 启动MinIO

执行以下命令启动MinIO
```bash
systemctl start minio
```
执行以下命令查询运行状态
```bash
systemctl status minio
```
设置MinIO开机自启
```bash
systemctl enable minio
```
访问MinIO管理页面

管理页面的访问地址为：http://192.168.10.101:9001

注意：

ip需要根据实际情况做出修改