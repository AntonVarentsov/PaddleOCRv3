Server snapshot (vast.ai)
=========================

Location
--------
- Host: 91.150.160.38
- SSH: `ssh -p 13664 root@91.150.160.38`
- Service: uvicorn `api.main:app` (started by `/app/onstart.sh`), listens on port 8000 (forwarded to 8080 locally in the provided SSH command).
- Project root on server: `/app`

Layout observed on server
-------------------------
- `/app/api/main.py`
- `/app/api/requirements.txt`
- `/app/requirements.txt`
- `/app/onstart.sh`
- `/app/scripts/test_gpu.py`
- `/app/ports.log` (runtime port info from vast.ai)

Runtime dependency snapshot
---------------------------
`pip freeze` taken on the server:

```
aistudio-sdk==0.3.8
albucore==0.0.13
albumentations==1.4.10
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.10.0
bce-python-sdk==0.9.55
beautifulsoup4==4.14.3
blinker==1.4
cachetools==6.2.2
certifi==2025.8.3
chardet==5.2.0
charset-normalizer==3.4.4
click==8.3.1
colorlog==6.10.1
contourpy==1.3.2
cryptography==3.4.8
cssselect==1.3.0
cssutils==2.11.1
cycler==0.12.1
Cython==3.2.2
dbus-python==1.2.18
distro==1.7.0
einops==0.8.1
et_xmlfile==2.0.0
exceptiongroup==1.3.0
fastapi==0.123.5
filelock==3.20.0
fire==0.7.1
fonttools==4.61.0
fsspec==2025.10.0
ftfy==6.3.1
future==1.0.0
h11==0.16.0
hf-xet==1.2.0
httpcore==1.0.9
httplib2==0.20.2
httpx==0.28.1
huggingface_hub==1.1.7
idna==3.10
ImageIO==2.37.2
imagesize==1.4.1
imgaug==0.4.0
importlib-metadata==4.6.4
jeepney==0.7.1
Jinja2==3.1.6
joblib==1.5.2
keyring==23.5.0
kiwisolver==1.4.9
launchpadlib==1.10.16
lazr.restfulclient==0.14.4
lazr.uri==1.0.6
lazy_loader==0.4
lmdb==1.7.5
lxml==6.0.2
MarkupSafe==3.0.3
matplotlib==3.10.7
modelscope==1.32.0
more-itertools==8.10.0
networkx==3.4.2
numpy==1.26.4
nvidia-cublas-cu11==11.11.3.6
nvidia-cuda-cupti-cu11==11.8.87
nvidia-cuda-nvrtc-cu11==11.8.89
nvidia-cuda-runtime-cu11==11.8.89
nvidia-cudnn-cu11==8.9.6.50
nvidia-cufft-cu11==10.9.0.58
nvidia-curand-cu11==10.3.0.86
nvidia-cusolver-cu11==11.4.1.48
nvidia-cusparse-cu11==11.7.5.86
nvidia-nccl-cu11==2.19.3
nvidia-nvtx-cu11==11.8.86
oauthlib==3.2.0
opencv-contrib-python==4.10.0.84
opencv-python==4.11.0.86
opencv-python-headless==4.11.0.86
openpyxl==3.1.5
opt-einsum==3.3.0
packaging==25.0
paddleocr==3.3.2
paddlepaddle-gpu==3.2.0
paddlex==3.3.10
pandas==2.3.3
pdf2image==1.17.0
pillow==11.2.1
premailer==3.10.0
prettytable==3.17.0
protobuf==6.32.0
psutil==7.1.3
py-cpuinfo==9.0.0
pyclipper==1.4.0
pycryptodome==3.23.0
pydantic==2.12.5
pydantic_core==2.41.5
PyGObject==3.42.1
PyJWT==2.3.0
pyparsing==3.2.5
pypdfium2==5.1.0
python-apt==2.4.0+ubuntu4
python-bidi==0.6.7
python-dateutil==2.9.0.post0
python-docx==1.2.0
python-multipart==0.0.20
pytz==2025.2
PyYAML==6.0.2
RapidFuzz==3.14.3
regex==2025.11.3
requests==2.32.5
ruamel.yaml==0.18.16
ruamel.yaml.clib==0.2.15
safetensors==0.6.2
scikit-image==0.25.2
scikit-learn==1.7.2
scipy==1.15.3
SecretStorage==3.3.1
sentencepiece==0.2.1
shapely==2.1.2
shellingham==1.5.4
six==1.17.0
sniffio==1.3.1
soupsieve==2.8
starlette==0.50.0
termcolor==3.2.0
threadpoolctl==3.6.0
tifffile==2025.5.10
tiktoken==0.12.0
tokenizers==0.22.1
tomli==2.3.0
tqdm==4.67.1
typer-slim==0.20.0
typing-inspection==0.4.2
typing_extensions==4.15.0
tzdata==2025.2
ujson==5.11.0
urllib3==2.5.0
uvicorn==0.38.0
wadllib==1.3.6
wcwidth==0.2.14
zipp==1.0.0
```

Notes
-----
- The service itself only needs the dependencies listed in `requirements.txt`; the full freeze above reflects everything that came with the current VM/image and PaddleOCR install.
- `paddlepaddle-gpu==3.2.0` with CUDA 11.8 is present on the server; GPU support should match that version when reproducing locally.
- Start command on the server is `bash onstart.sh` from `/app`.
