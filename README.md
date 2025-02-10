# convert2onnx
 
## 가상환경 진입 방법

```bash
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# 또는 Windows (CMD)
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

## 필요한 패키지 설치

```bash
pip install torch transformers onnx onnxruntime
```

## requirements.txt 만들기

```bash
pip freeze > requirements.txt
```

## requirements.txt install 하기

```bash
pip install -r requirements.txt
```