# DocParser Docker Deployment Guide

Bu rehber DocParser uygulamasını Docker ile hem API hem de Web Interface ile birlikte çalıştırmak için gerekli talimatları içerir.

## 🚀 Hızlı Başlangıç

### 1. Ön Gereksinimler

- Docker ve Docker Compose yüklü olmalı
- NVIDIA GPU ve CUDA desteği (model çalıştırmak için)
- En az 8GB GPU belleği

### 2. Model İndirme

Önce DotsOCR model ağırlıklarını indirin:

```bash
# Model indirme scripti ile
python tools/download_model.py

# Veya manuel olarak weights/ klasörüne
# https://huggingface.co/rednote-hilab/dots.ocr
```

### 3. Docker ile Başlatma

```bash
# Tüm servisleri başlat (önerilen)
make quick-start

# Veya manuel olarak
docker-compose -f docker/docker-compose.yml up -d
```

### 4. Erişim

- **Web Interface**: http://localhost:7860
- **API Documentation**: http://localhost:7860/docs  
- **vLLM Server**: http://localhost:9998

## 📋 Kullanılabilir Komutlar

### Make Komutları

```bash
make help           # Tüm komutları göster
make build          # Docker image'ı oluştur
make up             # Servisleri başlat
make down           # Servisleri durdur
make logs           # Logları görüntüle
make status         # Servis durumunu kontrol et
make health         # Sağlık kontrolü
make clean          # Temizlik yap
```

### Manuel Docker Komutları

```bash
# Build
docker build -t dotsocr -f docker/Dockerfile .

# Run services
docker-compose -f docker/docker-compose.yml up -d

# Check logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down
```

## 🔧 Yapılandırma

### Environment Variables

```bash
# Application settings
APP_MODE=combined        # combined|api|ui|separate
APP_HOST=0.0.0.0
APP_PORT=7860

# vLLM settings
VLLM_SERVER=vllm-server:9998
VLLM_PORT=9998
```

### Docker Compose Override

Özel ayarlar için `docker-compose.override.yml` oluşturun:

```yaml
version: '3.8'
services:
  dotsocr-app:
    environment:
      - APP_MODE=api  # Sadece API
    ports:
      - "9000:7860"   # Farklı port
```

## 🌐 Kullanım Modları

### 1. Combined Mode (Varsayılan)
Hem API hem Web Interface aynı portta:
```bash
python app.py --mode combined --port 7860
```
- Web UI: http://localhost:7860
- API: http://localhost:7860/api/*

### 2. API Only Mode  
Sadece REST API:
```bash
python app.py --mode api --port 7860
```
- API: http://localhost:7860/*

### 3. UI Only Mode
Sadece Web Interface:
```bash
python app.py --mode ui --port 7860
```
- Web UI: http://localhost:7860

### 4. Separate Mode
API ve UI ayrı portlarda:
```bash
python app.py --mode separate --api-port 7860 --ui-port 7861
```
- API: http://localhost:7860
- Web UI: http://localhost:7861

## 📊 API Kullanımı

### PDF Parsing

```bash
curl -X POST "http://localhost:7860/api/parse/pdf" \
  -F "file=@document.pdf" \
  -F "prompt_mode=prompt_layout_all_en"
```

### Image Parsing

```bash
curl -X POST "http://localhost:7860/api/parse/image" \
  -F "file=@image.jpg" \
  -F "prompt_mode=prompt_layout_all_en"
```

### Python Client

```python
import requests

# PDF parsing
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:7860/api/parse/pdf",
        files={"file": f},
        data={"prompt_mode": "prompt_layout_all_en"}
    )

result = response.json()
print(f"Found {result['total_elements']} elements")

# Download results
download_resp = requests.get(f"http://localhost:7860/api/download/{result['session_id']}")
with open("results.zip", "wb") as f:
    f.write(download_resp.content)
```

## 🔍 Monitoring ve Debugging

### Sağlık Kontrolü

```bash
# Uygulama durumu
curl http://localhost:7860/health

# vLLM durumu  
curl http://localhost:9998/health

# Make ile
make health
```

### Logları İzleme

```bash
# Tüm servisler
make logs

# Sadece uygulama
make logs-app

# Sadece vLLM
make logs-vllm
```

### Container'a Shell Erişimi

```bash
# Uygulama container'ı
make shell

# vLLM container'ı
make shell-vllm
```

## 🛠️ Geliştirme

### Local Development

```bash
# Geliştirme ortamı kurulumu
make setup-dev

# Local çalıştırma
make dev

# Sadece API
make dev-api

# Sadece UI
make dev-ui
```

### Test

```bash
# API testleri çalıştır
make test

# Manuel test
python test_api.py
```

## 🚨 Sorun Giderme

### Yaygın Sorunlar

1. **GPU Bellek Hatası**
   ```
   RuntimeError: CUDA out of memory
   ```
   - `gpu-memory-utilization` değerini düşürün (0.6-0.8)
   - Daha küçük batch size kullanın

2. **Model Bulunamadı**
   ```
   FileNotFoundError: weights/DotsOCR not found
   ```
   - Model dosyalarını doğru konuma indirin
   - Volume mount'ları kontrol edin

3. **Port Çakışması**
   ```
   Port 7860 already in use
   ```
   - Farklı port kullanın: `--port 7861`
   - Çakışan servisi durdurun

4. **vLLM Bağlantı Hatası**
   ```
   Connection refused to vllm server
   ```
   - vLLM servisinin çalıştığını kontrol edin
   - Network bağlantısını kontrol edin

### Log Analizi

```bash
# Detaylı loglar
docker-compose -f docker/docker-compose.yml logs --tail=100

# Belirli servise odaklan
docker logs dotsocr-app --follow
```

### Temizlik

```bash
# Geçici temizlik
make clean

# Tam temizlik
make clean-all
```

## 📈 Performans Optimizasyonu

### GPU Ayarları

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0,1  # Multi-GPU
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
          device_ids: ['0', '1']
```

### Bellek Optimizasyonu

```bash
# vLLM ayarları
--gpu-memory-utilization 0.8
--tensor-parallel-size 2    # Multi-GPU için
--max-model-len 8192        # Max sequence length
```

## 🔒 Güvenlik

### Production Deployment

1. **CORS Ayarları**
   ```python
   # api_server.py içinde
   allow_origins=["https://yourdomain.com"]
   ```

2. **Authentication** (opsiyonel)
   ```python
   # API key middleware ekleyin
   ```

3. **Rate Limiting**
   ```python
   # slowapi ile rate limiting
   ```

### Firewall

```bash
# Sadece gerekli portları açın
ufw allow 7860  # API + UI
ufw allow 9998  # vLLM
```

## 📞 Destek

- **GitHub Issues**: Sorunları raporlayın
- **Documentation**: API docs http://localhost:7860/docs
- **Examples**: `api/client_example.py` dosyasını inceleyin
