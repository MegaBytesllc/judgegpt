FROM python:3.12-slim

WORKDIR /app

# Install Docker CLI via direct download from Docker's official CDN.
# This avoids apt mirror CDN issues (403 Forbidden on Fastly IPs) seen on some networks.
# The Python urllib is already available in the base image and supports HTTPS.
# Supports x86_64 (Intel/AMD) and arm64 (Apple Silicon, AWS Graviton, Raspberry Pi).
RUN python3 -c "\
import urllib.request, tarfile, os, platform; \
arch = {'x86_64':'x86_64','aarch64':'aarch64','arm64':'aarch64'}.get(platform.machine(),'x86_64'); \
url = 'https://download.docker.com/linux/static/stable/' + arch + '/docker-27.5.1.tgz'; \
print('Downloading Docker CLI for', arch, '...'); \
urllib.request.urlretrieve(url, '/tmp/d.tgz'); \
tf = tarfile.open('/tmp/d.tgz'); \
m = next(m for m in tf.getmembers() if m.name.endswith('/docker')); \
m.name = 'docker'; \
tf.extract(m, '/usr/local/bin'); \
tf.close(); \
os.chmod('/usr/local/bin/docker', 0o755); \
os.remove('/tmp/d.tgz')" \
    && docker --version

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
