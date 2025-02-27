import torch
import os
from transformers import AutoTokenizer, AutoModel
from transformers.models.llama.modeling_llama import LlamaModel

# 원본 forward 메서드 저장
original_forward = LlamaModel.forward

# ONNX 변환용 forward 메서드 오버라이드
def custom_forward(self, *args, **kwargs):
    # past_key_values와 use_cache 매개변수 제거
    kwargs.pop('past_key_values', None)
    kwargs['use_cache'] = False
    
    # 원본 forward 호출하되 last_hidden_state만 반환
    outputs = original_forward(self, *args, **kwargs)
    return outputs.last_hidden_state

# 임시로 forward 메서드 변경
LlamaModel.forward = custom_forward

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

        print(f"\n3. ONNX 변환 중...")
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "last_hidden_state": {0: "batch_size", 1: "sequence_length"}
            }
        )
        print(f"[v] ONNX 변환 완료: {output_path}")

    except Exception as e:
        print(f"\n[x] 오류 발생: {str(e)}")
        raise
    
    finally:
        # 원본 forward 메서드 복원
        LlamaModel.forward = original_forward

if __name__ == "__main__":
    MODEL_NAME = "kakaocorp/kanana-nano-2.1b-embedding"        
    ONNX_PATH = "./oonx/kanana-nano-2.1b-embedding/kanana-nano-2.1b-embedding.onnx"
    
    convert_model_to_onnx(MODEL_NAME, ONNX_PATH)