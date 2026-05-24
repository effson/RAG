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