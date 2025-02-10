import torch
from transformers import AutoTokenizer, AutoModel

def convert_model_to_onnx(
    model_name: str,
    output_path: str,
    example_text: str = "이 문장은 ONNX 변환 테스트용 예시입니다.",
    opset_version: int = 14
):
    """
    Hugging Face 모델을 ONNX 형식으로 변환하는 함수

    Args:
        model_name (str): Hugging Face 모델 이름 (예: "skt/kobert-base-v1")
        output_path (str): ONNX 모델을 저장할 경로
        example_text (str): 변환 시 사용할 예시 문장
        opset_version (int): ONNX opset 버전
    """
    try:
        print(f"\n1. 모델 및 토크나이저 로드 중... ({model_name})")
        # 토크나이저 설정
        # - trust_remote_code: 원격 코드 신뢰 여부 (커스텀 코드 포함된 모델용)
        # - use_fast: Fast 토크나이저 사용 여부 (False로 설정하여 호환성 향상)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=True
        )

        # 모델 설정 및 평가 모드로 전환
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()  # 평가 모드 설정 (학습 모드 비활성화)
        print("✓ 모델 및 토크나이저 로드 완료")

        print(f"\n2. 예시 입력 생성 중...")
        # 토큰화 및 PyTorch 텐서로 변환
        inputs = tokenizer(example_text, return_tensors="pt")
        print(f"✓ 입력 생성 완료 (토큰 수: {len(inputs['input_ids'][0])})")

        print(f"\n3. ONNX 변환 중...")
        # ONNX 모델 변환 설정
        torch.onnx.export(
            model,                                      # 변환할 PyTorch 모델
            (inputs["input_ids"], inputs["attention_mask"]),  # 모델 입력값
            output_path,                                # 저장 경로
            export_params=True,                         # 모델 파라미터 포함
            opset_version=opset_version,                # ONNX 버전
            do_constant_folding=True,                   # 상수 최적화 사용
            input_names=["input_ids", "attention_mask"],# 입력 레이어 이름
            output_names=["output"],                    # 출력 레이어 이름
            dynamic_axes={                              # 가변 크기 설정
                # 배치 크기(batch_size)와 문장 길이(sequence_length)를 동적으로 설정
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "output": {0: "batch_size"}
            }
        )
        print(f"✓ ONNX 변환 완료: {output_path}")

    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 변환 설정
    MODEL_NAME = "skt/kobert-base-v1"        # 변환할 모델 이름
    ONNX_PATH = "./oonx/kobert-base-v1.onnx" # 저장할 경로
    
    # 변환 실행
    convert_model_to_onnx(MODEL_NAME, ONNX_PATH)