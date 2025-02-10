import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

def load_and_test_onnx_model(onnx_path: str, model_name: str, test_text: str):
    """
    ONNX 모델을 로드하고 테스트하는 함수
    
    Args:
        onnx_path (str): ONNX 모델 파일 경로
        model_name (str): Hugging Face 모델 이름
        test_text (str): 테스트할 입력 텍스트
    """
    try:
        # 1. ONNX 모델 로드
        print(f"\n1. ONNX 모델 로드 중... ({onnx_path})")
        session = ort.InferenceSession(onnx_path)
        print("✓ 모델 로드 성공")

        # 2. 토크나이저 로드
        print(f"\n2. 토크나이저 로드 중... ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("✓ 토크나이저 로드 성공")

        # 3. 입력 텍스트 토큰화
        print(f"\n3. 입력 텍스트 토큰화 중...")
        print(f"입력 텍스트: '{test_text}'")
        inputs = tokenizer(test_text, return_tensors="np")
        
        # 토큰화 결과 상세 출력
        tokens = tokenizer.tokenize(test_text)
        print(f"토큰화 결과: {tokens}")
        print(f"토큰 수: {len(tokens)}")

        # 4. 모델 입력 준비
        print("\n4. 모델 입력 준비 중...")
        ort_inputs = {
            session.get_inputs()[0].name: inputs["input_ids"],
            session.get_inputs()[1].name: inputs["attention_mask"]
        }

        # 5. 모델 추론 실행
        print("\n5. 모델 추론 실행 중...")
        ort_outputs = session.run(None, ort_inputs)

        # 6. 결과 분석
        print("\n6. 모델 출력 분석:")
        print("\n[모델 입력 정보]")
        for input_meta in session.get_inputs():
            print(f"- 입력 이름: {input_meta.name}")
            print(f"  shape: {input_meta.shape}")
            print(f"  타입: {input_meta.type}")

        print("\n[모델 출력 정보]")
        for i, output in enumerate(ort_outputs):
            print(f"\n출력 {i+1}번:")
            print(f"- Shape: {output.shape}")
            print(f"- 설명:")
            if i == 0:
                print("  * 각 토큰별 임베딩 벡터")
                print(f"  * (배치 크기: {output.shape[0]}, 토큰 수: {output.shape[1]}, 임베딩 차원: {output.shape[2]})")
            else:
                print("  * 전체 문장의 임베딩 벡터")
                print(f"  * (배치 크기: {output.shape[0]}, 임베딩 차원: {output.shape[1]})")
            
            # 기본적인 통계 정보 출력
            print(f"- 통계:")
            print(f"  * 평균값: {np.mean(output):.4f}")
            print(f"  * 표준편차: {np.std(output):.4f}")
            print(f"  * 최소값: {np.min(output):.4f}")
            print(f"  * 최대값: {np.max(output):.4f}")

        print("\n✓ 검증 완료: 모델이 정상적으로 작동합니다.")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    # 테스트 설정
    ONNX_MODEL_PATH = "./oonx/kobert-base-v1.onnx"
    MODEL_NAME = "skt/kobert-base-v1"
    TEST_TEXT = "이 모델은 한국어 자연어 처리를 위한 ONNX 변환 모델입니다."

    # 테스트 실행
    load_and_test_onnx_model(ONNX_MODEL_PATH, MODEL_NAME, TEST_TEXT)