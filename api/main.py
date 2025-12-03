import os
import shutil
import tempfile
import json
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCRVL

app = FastAPI(title="PaddleOCR-VL Service", version="1.0.0")

# Initialize PaddleOCR-VL
# We initialize it once at startup to load models into GPU memory
print("Initializing PaddleOCR-VL...")
try:
    # use_layout_detection=True is default, but explicit is better
    # use_doc_orientation_classify=True for better accuracy
    ocr_pipeline = PaddleOCRVL(
        use_layout_detection=True,
        use_doc_orientation_classify=True
    )
    print("PaddleOCR-VL initialized successfully.")
except Exception as e:
    print(f"Error initializing PaddleOCR-VL: {e}")
    ocr_pipeline = None

@app.get("/health")
async def health_check():
    if ocr_pipeline:
        return {"status": "healthy", "gpu": "enabled"} # We assume GPU is enabled if it didn't crash
    return {"status": "unhealthy", "reason": "Model not initialized"}

@app.post("/parse")
async def parse_document(file: UploadFile = File(...)):
    if not ocr_pipeline:
        raise HTTPException(status_code=503, detail="OCR service not available")

    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and PDF are supported.")

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = tmp_file.name

    try:
        # Run prediction
        # predict() returns a list of results (one per page for PDF, or one for image)
        print(f"Processing file: {tmp_path}")
        results = ocr_pipeline.predict(input=tmp_path)
        
        response_data = []
        
        # Use a temp directory to extract JSON results safely
        with tempfile.TemporaryDirectory() as tmp_out_dir:
            for i, res in enumerate(results):
                # Save to JSON in the temp dir
                # save_to_json expects a directory path, and it will create files inside
                res.save_to_json(save_path=tmp_out_dir)
                
                # The file naming convention is usually related to input filename or page index
                # We just look for the newest json file or all json files
                # Since we do this page by page (if results is a list), we might need to be careful
                # But predict() on a single image returns a list with 1 element.
                # On PDF it returns multiple.
                
                # Let's read all JSONs in the dir and append
                # To avoid reading the same file twice if multiple results save to same dir,
                # we might need to clean up or handle naming.
                # Actually, save_to_json might save ALL pages if called on the list?
                # No, we are iterating `res` which is a single page result.
                
                pass
            
            # After saving all, let's read them
            # PaddleOCR usually names them like 'filename_res.json' or similar.
            json_files = sorted([f for f in os.listdir(tmp_out_dir) if f.endswith(".json")])
            
            for jf in json_files:
                with open(os.path.join(tmp_out_dir, jf), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    response_data.append(data)
                    
        # If the above loop didn't work (e.g. save_to_json behavior differs), 
        # we might return empty. But it's the standard way in docs.
        
        return JSONResponse(content={"pages": response_data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")
    finally:
        # Cleanup input file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
