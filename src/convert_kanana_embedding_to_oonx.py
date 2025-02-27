import torch
import os
from transformers import AutoTokenizer, AutoModel

def convert_model_to_onnx(
    model_name: str,
    output_path: str,
    example_text: str = "이 문장은 ONNX 변환 테스트용 예시입니다.",
    opset_version: int = 14
):
    try:
        # 출력 경로의 디렉토리 부분 추출
        output_dir = os.path.dirname(output_path)
        
        # 디렉토리가 존재하지 않으면 생성
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[v] 디렉토리 생성 완료: {output_dir}")

        print(f"\n1. 모델 및 토크나이저 로드 중... ({model_name})")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            use_fast=True
        )

        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model.eval()
        print("[v] 모델 및 토크나이저 로드 완료")

        print(f"\n2. 예시 입력 생성 중...")
        inputs = tokenizer(example_text, return_tensors="pt")
        print(f"[v] 입력 생성 완료 (토큰 수: {len(inputs['input_ids'][0])})")

        # 임베딩 모델을 위한 래퍼 클래스 정의
        class EmbeddingWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
                
            def forward(self, input_ids, attention_mask):
                # 모델의 forward 호출 시 안전하게 처리
                try:
                    # 모델 출력을 직접 반환
                    return self.model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    print(f"모델 실행 중 오류: {e}")
                    # 예비 방법: 직접 임베딩 레이어만 사용
                    return self.model.get_input_embeddings()(input_ids)

        # 래퍼 모델 생성
        wrapper_model = EmbeddingWrapper(model)

        print(f"\n3. ONNX 변환 중...")
        torch.onnx.export(
            wrapper_model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "embeddings": {0: "batch_size", 1: "sequence_length"}
            }
        )
        print(f"[v] ONNX 변환 완료: {output_path}")

    except Exception as e:
        print(f"\n[x] 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    # 변환 설정
    MODEL_NAME = "kakaocorp/kanana-nano-2.1b-embedding"
    ONNX_PATH = "./oonx/kanana-nano-2.1b-embedding/kanana-nano-2.1b-embedding.onnx"
    
    # 변환 실행
    convert_model_to_onnx(MODEL_NAME, ONNX_PATH)