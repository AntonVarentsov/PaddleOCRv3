# PaddleOCR-VL API Service

This project provides a Dockerized REST API service for PaddleOCR-VL (v3.x), supporting OCR and layout analysis (tables, blocks) with GPU acceleration.

## Requirements

- **NVIDIA GPU** with Compute Capability >= 7.0 (e.g., T4, RTX 20xx/30xx/40xx, A100).
- **NVIDIA Drivers**: Compatible with CUDA 11.8 (Driver version >= 450.80.02 for Linux).
- **Docker** >= 19.03.
- **NVIDIA Container Toolkit**: Installed and configured to allow Docker to access GPUs.

## Setup & Deployment

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2.  **Start the service**:
    ```bash
    docker compose up -d
    ```
    *Note: The first run will download the Docker image and PaddleOCR models (approx. several GBs). Models are persisted in the `models/` directory.*

3.  **Check logs**:
    ```bash
    docker compose logs -f
    ```

## Usage

### API Endpoint: `/parse`

Accepts an image (JPEG/PNG) or PDF file and returns structured layout and OCR results.

**Request:**
- Method: `POST`
- URL: `http://localhost:8000/parse`
- Body: `multipart/form-data` with key `file`.

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/parse" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/document.png"
```

**Response (JSON):**
```json
{
  "pages": [
    {
      "regions": [
        {
          "type": "text",
          "bbox": [100, 100, 200, 150],
          "res": { "text": "Sample Text", "confidence": 0.98 }
        },
        {
          "type": "table",
          "bbox": [50, 200, 500, 400],
          "res": { "html": "<table>...</table>" }
        }
      ]
    }
  ]
}
```

### Health Check

**Request:**
`GET /health`

**Response:**
`{"status": "healthy", "gpu": "enabled"}`

## Verification

To verify GPU support inside the container:

```bash
docker compose exec ocr-service python scripts/test_gpu.py
```

Output should confirm:
- `PaddlePaddle is compiled with CUDA.`
- `GPU detected successfully.`
- `PaddleOCR-VL initialized successfully.`

## Configuration

- **Port**: Default is `8000`. Change in `docker-compose.yml`.
- **GPU**: Default uses `device_ids: ["0"]`. Update `docker-compose.yml` for multiple GPUs.

## Publishing Docker Images

### Option 1: GitHub Container Registry (GHCR) - Automatic via CI/CD

The repository includes a GitHub Actions workflow that automatically builds and publishes images to GHCR.

**Setup:**
1. Push code to the `main` branch or create a version tag (e.g., `v1.0.0`).
2. GitHub Actions will automatically build and push the image.
3. After the first build, go to your GitHub profile → Packages → `paddleocrv3` → Package settings → Change visibility to **Public** (if desired).

**Using the published image:**
```bash
docker pull ghcr.io/<your-username>/paddleocrv3:latest
```

**Available tags:**
- `latest` - Latest build from main branch
- `v1.0.0` - Specific version tag
- `<commit-sha>` - Specific commit

### Option 2: Docker Hub - Manual Publishing

To build and push the image to Docker Hub manually:

1.  **Login to Docker Hub**:
    ```bash
    docker login
    ```

2.  **Build the image**:
    Replace `your_username` with your Docker Hub username.
    ```bash
    docker build -t your_username/paddleocr-vl-service:latest .
    ```

3.  **Push the image**:
    ```bash
    docker push your_username/paddleocr-vl-service:latest
    ```

### Local Testing Before Publishing

Test the image locally before publishing:
```bash
docker build -t paddleocrv3:local .
docker run --rm -p 8000:8000 paddleocrv3:local
```
