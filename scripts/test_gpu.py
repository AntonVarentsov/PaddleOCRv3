import paddle
import paddleocr
from paddleocr import PaddleOCRVL
import os

def check_gpu():
    print("Checking GPU availability...")
    try:
        paddle.utils.run_check()
    except Exception as e:
        print(f"paddle.utils.run_check() failed: {e}")
    
    if paddle.is_compiled_with_cuda():
        print("PaddlePaddle is compiled with CUDA.")
    else:
        print("WARNING: PaddlePaddle is NOT compiled with CUDA.")
        
    places = paddle.device.get_all_places()
    print(f"Available places: {places}")
    
    if not any(isinstance(p, paddle.CUDAPlace) for p in places):
        print("WARNING: No GPU detected by PaddlePaddle.")
    else:
        print("GPU detected successfully.")

def check_ocr_vl():
    print("\nChecking PaddleOCR-VL initialization...")
    try:
        # Initialize with minimal options to test loading
        pipeline = PaddleOCRVL(use_layout_detection=True, use_doc_orientation_classify=True)
        print("PaddleOCR-VL initialized successfully.")
    except Exception as e:
        print(f"Error initializing PaddleOCR-VL: {e}")

if __name__ == "__main__":
    check_gpu()
    check_ocr_vl()
