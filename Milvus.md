#  下载
```bash
wget https://github.com/milvus-io/milvus/releases/download/v2.4.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
```
# 启动
```bash
docker compose up -d
```
# 检查运行状态
```bash
docker compose ps
```

# 解决minio 端口冲突

```bash
vim ~/milvus/docker-compose.yml
```
找到minio 端口修改：
9001 -> 9003
9000 -> 9002