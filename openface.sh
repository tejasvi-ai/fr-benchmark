cd openface/models
bash models/get-models.sh
cd ..
docker build -t openface .
docker run -p 9000:9000 -p 8000:8000 -t -i openface /bin/bash
cd /root/openface