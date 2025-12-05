import os
import shutil
import tempfile
import json
import time
import gc
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, HTMLResponse
from paddleocr import PaddleOCR
from PIL import Image, ImageOps
from pdf2image import convert_from_path
from collections import defaultdict
import paddle

app = FastAPI(title="PaddleOCR with Tables Service")

def detect_tables_heuristic(blocks: List[Dict], img_width: int, img_height: int,
                            threshold: float = 0.02) -> List[Dict]:
    """Detect table-like structures from OCR text blocks using geometric heuristics."""
    if not blocks:
        return []

    # Extract block positions
    block_info = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        x_coords = [v['x'] for v in vertices]
        y_coords = [v['y'] for v in vertices]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        block_info.append({
            'block': block,
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'x_center': x_center, 'y_center': y_center,
            'width': x_max - x_min, 'height': y_max - y_min
        })

    # Find aligned rows
    y_threshold = threshold
    rows = []
    used_blocks = set()

    for i, info in enumerate(block_info):
        if i in used_blocks:
            continue
        row = [info]
        used_blocks.add(i)
        for j, other in enumerate(block_info):
            if j in used_blocks:
                continue
            if abs(info['y_center'] - other['y_center']) < y_threshold:
                row.append(other)
                used_blocks.add(j)
        if len(row) >= 2:
            rows.append(sorted(row, key=lambda x: x['x_center']))

    if len(rows) < 2:
        return []

    rows = sorted(rows, key=lambda r: r[0]['y_center'])

    # Find table candidates
    tables = []
    current_table_rows = []

    for row in rows:
        if not current_table_rows:
            current_table_rows.append(row)
            continue

        prev_row = current_table_rows[-1]
        y_gap = row[0]['y_center'] - prev_row[0]['y_center']

        if y_gap < 0.1:
            column_match = False
            for prev_block in prev_row:
                for curr_block in row:
                    if abs(prev_block['x_center'] - curr_block['x_center']) < threshold:
                        column_match = True
                        break
                if column_match:
                    break

            if column_match and len(row) >= 2:
                current_table_rows.append(row)
            else:
                if len(current_table_rows) >= 3:
                    tables.append(current_table_rows)
                current_table_rows = [row]
        else:
            if len(current_table_rows) >= 3:
                tables.append(current_table_rows)
            current_table_rows = [row]

    if len(current_table_rows) >= 3:
        tables.append(current_table_rows)

    # Create table regions
    table_regions = []
    for idx, table_rows in enumerate(tables):
        all_x = []
        all_y = []
        for row in table_rows:
            for block_info in row:
                all_x.extend([block_info['x_min'], block_info['x_max']])
                all_y.extend([block_info['y_min'], block_info['y_max']])

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        padding = 0.01
        x_min = max(0, x_min - padding)
        x_max = min(1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(1, y_max + padding)

        table_regions.append({
            'id': f'table-heuristic-{idx + 1}',
            'type': 'table',
            'confidence': 0.85,
            'rows': len(table_rows),
            'boundingPoly': {
                'normalizedVertices': [
                    {'x': x_min, 'y': y_min},
                    {'x': x_max, 'y': y_min},
                    {'x': x_max, 'y': y_max},
                    {'x': x_min, 'y': y_max}
                ]
            }
        })

    return table_regions


def convert_pdf_to_images(pdf_path: str, max_dim: int = 10000, dpi: int = 300) -> List[str]:
    """
    Convert a PDF into resized PNG images stored in temp files.
    Ensures the longest side of each page is below max_dim to reduce GPU memory use.
    """
    try:
        images = convert_from_path(pdf_path, dpi=dpi, fmt="png", thread_count=2)
    except Exception as e:
        raise HTTPException(500, f"PDF conversion failed: {e}")

    image_files = []
    for idx, img in enumerate(images, start=1):
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_size = (int(w * scale), int(h * scale))
            img = img.resize(new_size, Image.LANCZOS)

        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{idx}.png")
        img.save(tmp_img.name, format="PNG")
        image_files.append(tmp_img.name)

    return image_files

print("Initializing PaddleOCR...")
try:
    ocr = PaddleOCR(
        use_textline_orientation=False,  # keep only core det+rec to speed up
        lang="en",
        rec_batch_num=4,  # lower batch to avoid GPU OOM spikes
        det_limit_side_len=10000,
        det_limit_type="max"
    )
    print("PaddleOCR initialized!")
    # Warm up once so models are fully loaded in memory before first request
    tmp_path = None
    try:
        print("Running OCR warmup...")
        warmup_img = Image.new("RGB", (128, 128), color="white")
        fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # close immediately so Pillow can write on Windows
        warmup_img.save(tmp_path, format="PNG")
        ocr.predict(input=tmp_path)
        print("OCR warmup completed.")
    except Exception as warmup_err:
        print(f"OCR warmup failed: {warmup_err}")
    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
except Exception as e:
    print(f"OCR Error: {e}")
    import traceback; traceback.print_exc()
    ocr = None

# Table layout pipeline disabled to reduce GPU memory use
table_pipeline = None

@app.get("/health")
async def health():
    return {
        "status": "healthy" if ocr else "unhealthy",
        "ocr": ocr is not None,
        "tables": False
    }

def process_image_file(image_path: str, detect_tables: bool, page_offset: int = 1) -> List[Dict]:
    """Run OCR + optional table detection on a single image file."""
    PAD = None
    padded_path = None

    # Pad image on all sides to avoid clipping on edges (and handle rotations)
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size
        # Stronger dynamic padding to preserve left/top content on small renders
        PAD = max(100, min(240, int(max(orig_w, orig_h) * 0.04)))
        padded_img = ImageOps.expand(img, border=PAD, fill="white")
        padded_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        padded_img.save(padded_path, format="PNG")
        pad_w, pad_h = padded_img.size

    try:
        page_start = time.perf_counter()
        result = ocr.predict(input=padded_path)
        print(f"OCR page done in {time.perf_counter() - page_start:.2f}s")
    finally:
        if padded_path and os.path.exists(padded_path):
            try:
                os.remove(padded_path)
            except Exception:
                pass
        # Release GPU cache between pages
        try:
            paddle.device.cuda.synchronize()
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

    pages = []
    for page_idx, page_result in enumerate(result, start=page_offset):
        if not page_result:
            continue

        with tempfile.TemporaryDirectory() as td:
            page_result.save_to_json(td)
            json_files = [f for f in os.listdir(td) if f.endswith(".json")]

            if not json_files:
                continue

            with open(os.path.join(td, json_files[0]), 'r', encoding='utf-8') as f:
                raw_data = json.load(f)

            rec_texts = raw_data.get('rec_texts', [])
            rec_scores = raw_data.get('rec_scores', [])
            rec_polys = raw_data.get('rec_polys', [])

            blocks = []
            for i, (text, score, poly) in enumerate(zip(rec_texts, rec_scores, rec_polys)):
                if not text.strip():
                    continue

                normalized_vertices = []
                for point in poly:
                    adj_x = max(0.0, min(orig_w, point[0] - PAD))
                    adj_y = max(0.0, min(orig_h, point[1] - PAD))
                    normalized_vertices.append({
                        "x": adj_x / orig_w if orig_w else 0.0,
                        "y": adj_y / orig_h if orig_h else 0.0
                    })

                blocks.append({
                    "id": f"{page_idx}-{i + 1}",
                    "type": "text",
                    "text": text,
                    "confidence": float(score),
                    "boundingPoly": {
                        "normalizedVertices": normalized_vertices
                    }
                })

            page_data = {
                "page": page_idx,
                "width": orig_w,
                "height": orig_h,
                "blocks": blocks
            }

            if detect_tables:
                try:
                    heuristic_tables = detect_tables_heuristic(blocks, orig_w, orig_h)
                    if heuristic_tables:
                        page_data.setdefault('tables', []).extend(heuristic_tables)
                        print(f"Heuristic found {len(heuristic_tables)} table(s)")
                except Exception as e:
                    print(f"Heuristic table detection error: {e}")
                    import traceback; traceback.print_exc()

            pages.append(page_data)

    return pages


@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>PaddleOCR UI</title>
  <style>
    :root { color-scheme: light; font-family: "Inter", system-ui, -apple-system, sans-serif; }
    body { margin: 0; padding: 24px; background: #f6f7fb; color: #111; }
    h1 { margin: 0 0 16px; }
    .card { background: #fff; border: 1px solid #e3e5ea; border-radius: 12px; padding: 20px; max-width: 960px; box-shadow: 0 10px 40px rgba(17, 24, 39, 0.06); }
    .row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }
    label { font-weight: 600; font-size: 14px; }
    input[type="file"] { margin-top: 6px; }
    input[type="number"] { width: 120px; padding: 6px 8px; border: 1px solid #cdd1d8; border-radius: 8px; }
    button { background: #111827; color: #fff; border: none; padding: 10px 16px; border-radius: 10px; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { margin-top: 12px; font-size: 14px; color: #374151; }
    textarea { width: 100%; height: 420px; margin-top: 16px; padding: 12px; border: 1px solid #cdd1d8; border-radius: 10px; font-family: "SFMono-Regular", Consolas, monospace; font-size: 13px; background: #0f172a; color: #e5e7eb; }
  </style>
</head>
<body>
  <div class="card">
    <h1>PaddleOCR Upload</h1>
    <form id="uploadForm">
      <div class="row">
        <div>
          <label for="file">File (PDF/PNG/JPG)</label><br/>
          <input id="file" name="file" type="file" accept=".pdf,image/png,image/jpeg" required />
        </div>
        <div>
          <label for="max_dim">Max side (px)</label><br/>
          <input id="max_dim" name="max_dim" type="number" value="10000" min="256" max="12000" />
        </div>
        <div>
          <label for="dpi">DPI (PDF)</label><br/>
          <input id="dpi" name="dpi" type="number" value="300" min="72" max="600" />
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:22px;">
          <input id="detect_tables" type="checkbox" name="detect_tables" />
          <label for="detect_tables" style="margin:0;font-weight:500;">Detect tables</label>
        </div>
        <div style="flex:1"></div>
        <div>
          <button id="submitBtn" type="submit">Run OCR</button>
        </div>
      </div>
    </form>
    <div id="status" class="status"></div>
    <textarea id="output" readonly placeholder="Response JSON will appear here"></textarea>
  </div>
  <script>
    const form = document.getElementById("uploadForm");
    const statusEl = document.getElementById("status");
    const output = document.getElementById("output");
    const submitBtn = document.getElementById("submitBtn");

    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById("file");
      if (!fileInput.files.length) {
        statusEl.textContent = "Please choose a file.";
        return;
      }
      const maxDim = document.getElementById("max_dim").value || "10000";
      const dpi = document.getElementById("dpi").value || "300";
      const detectTables = document.getElementById("detect_tables").checked ? "true" : "false";

      const data = new FormData();
      data.append("file", fileInput.files[0]);

      const url = `/parse?detect_tables=${detectTables}&max_dim=${encodeURIComponent(maxDim)}&dpi=${encodeURIComponent(dpi)}`;

      statusEl.textContent = "Processing...";
      submitBtn.disabled = true;
      output.value = "";

      try {
        const started = performance.now();
        const res = await fetch(url, { method: "POST", body: data });
        const text = await res.text();
        try {
          const json = JSON.parse(text);
          output.value = JSON.stringify(json, null, 2);
          const duration = (json.time_sec ?? ((performance.now() - started) / 1000)).toFixed(2);
          statusEl.textContent = res.ok ? `Done in ${duration}s` : `Error (${res.status})`;
        } catch {
          output.value = text;
          statusEl.textContent = res.ok ? "Done (non-JSON response)" : `Error (${res.status})`;
        }
      } catch (err) {
        statusEl.textContent = `Request failed: ${err}`;
      } finally {
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
    """


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    detect_tables: bool = Query(False),
    max_dim: int = Query(10000, description="Max image side (px) after PDF conversion"),
    dpi: int = Query(300, description="DPI for PDF to image conversion")
):
    if not ocr:
        raise HTTPException(503, "OCR unavailable")

    start_time = time.perf_counter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    converted_images: List[str] = []
    all_pages: List[Dict] = []

    try:
        is_pdf = (file.content_type == "application/pdf") or tmp_path.lower().endswith(".pdf")
        inputs = [tmp_path]

        if is_pdf:
            print(f"Converting PDF to images: {tmp_path}, max_dim={max_dim}, dpi={dpi}")
            converted_images = convert_pdf_to_images(tmp_path, max_dim=max_dim, dpi=dpi)
            inputs = converted_images

        page_counter = 1
        for image_path in inputs:
            print(f"Processing image: {image_path}")
            pages = process_image_file(image_path, detect_tables, page_counter)
            all_pages.extend(pages)
            page_counter += len(pages)

        duration = time.perf_counter() - start_time
        return JSONResponse(content={"pages": all_pages, "time_sec": round(duration, 3)})

    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, f"Error: {e}")
    finally:
        for path in [tmp_path] + converted_images:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        # Force garbage collection and clear CUDA cache after each request
        try:
            gc.collect()
            paddle.device.cuda.synchronize()
            paddle.device.cuda.empty_cache()
        except Exception:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
