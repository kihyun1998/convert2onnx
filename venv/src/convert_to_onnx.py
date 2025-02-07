import torch
from transformers import AutoTokenizer, AutoModel

# 모델 이름 (Hugging Face 모델 페이지의 경로)
model_name = "Dongjin-kr/ko-reranker"

# 토크나이저와 모델 로드 (커스텀 코드가 있을 경우 trust_remote_code 옵션 추가)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
model.eval()

# 더미 입력 데이터 생성 (예시 문장을 이용)
example_text = "이 문장은 ONNX 변환 테스트용 예시입니다."
inputs = tokenizer(example_text, return_tensors="pt")

# ONNX 모델 파일 경로 지정
onnx_model_path = "./oonx/ko_reranker.onnx"

# torch.onnx.export 함수를 사용하여 모델을 ONNX 포맷으로 변환
torch.onnx.export(
    model,                                  # 변환할 모델
    (inputs["input_ids"], inputs["attention_mask"]),  # 모델의 입력 (필요시 token_type_ids 등 추가)
    onnx_model_path,                        # 저장할 ONNX 파일 경로
    export_params=True,                     # 모델의 학습된 파라미터 포함
    opset_version=14,                       # 사용하려는 ONNX opset 버전
    do_constant_folding=True,               # 상수 폴딩 최적화 적용
    input_names=["input_ids", "attention_mask"],  # 입력 텐서 이름
    output_names=["output"],                # 출력 텐서 이름 (모델에 따라 다를 수 있음)
    dynamic_axes={                          # 동적 배치 크기와 시퀀스 길이 지정
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size"}
    }
)

print(f"ONNX 모델이 {onnx_model_path}에 저장되었습니다.")