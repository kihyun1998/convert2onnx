import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# 1. ONNX 모델 파일 경로 (메인 .onnx 파일과 외부 데이터 파일들이 모두 같은 폴더에 있어야 함)
onnx_model_path = "./oonx/ko_reranker.onnx"

# 2. ONNX Runtime 세션 생성 (모델 로드)
session = ort.InferenceSession(onnx_model_path)

# 3. 동일한 모델 이름을 사용해 토크나이저 로드 (추론용 입력 생성)
model_name = "Dongjin-kr/ko-reranker"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 4. 테스트용 예시 문장 (모델이 처리할 입력 텍스트)
example_text = "이 문장은 ONNX 모델의 로드를 검증하기 위한 예시입니다."

# 5. 토크나이저를 이용해 입력 데이터 생성 (numpy 배열 형식으로 변환)
#    - input_ids와 attention_mask가 모델의 입력으로 사용됨
inputs = tokenizer(example_text, return_tensors="np")

# 6. ONNX Runtime 세션에서 요구하는 입력 딕셔너리 생성
#    - session.get_inputs()를 통해 입력 이름을 가져오므로, 모델에 맞는 이름으로 입력을 전달함
ort_inputs = {
    session.get_inputs()[0].name: inputs["input_ids"],
    session.get_inputs()[1].name: inputs["attention_mask"]
}

# 7. 모델 추론 수행 (모든 출력 값을 리스트로 반환)
ort_outs = session.run(None, ort_inputs)

# 8. 출력 결과 확인 (각 출력 텐서의 shape를 출력)
print("ONNX 모델 출력:")
for i, output in enumerate(ort_outs):
    print(f"Output {i} shape: {output.shape}")
