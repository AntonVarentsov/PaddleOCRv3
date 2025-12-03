from paddleocr import TableRecognitionPipelineV2
import json
import tempfile
import os

print("Initializing table pipeline...")
table_pipeline = TableRecognitionPipelineV2(use_layout_detection=True, use_ocr_model=True)

print("Running prediction...")
result = table_pipeline.predict(input="/tmp/test_image.png")

for idx, res in enumerate(result):
    print(f"\nPage {idx}:")

    with tempfile.TemporaryDirectory() as td:
        res.save_to_json(td)
        json_files = [f for f in os.listdir(td) if f.endswith(".json")]

        if json_files:
            with open(os.path.join(td, json_files[0])) as f:
                data = json.load(f)
                print(f"  Keys: {list(data.keys())}")

                if "layout_res" in data:
                    layout_count = len(data["layout_res"])
                    print(f"  Layout results: {layout_count} items")
                    for i, layout in enumerate(data["layout_res"][:10]):
                        label = layout.get("label")
                        score = layout.get("score", 0)
                        bbox = layout.get("bbox", [])
                        print(f"    {i}: label={label}, score={score:.3f}, bbox={bbox}")

                if "table_results" in data:
                    table_count = len(data["table_results"])
                    print(f"  Table results: {table_count} items")

print("\nSaving full result to debug_table_result.json...")
result = table_pipeline.predict(input="/tmp/test_image.png")
for idx, res in enumerate(result):
    with tempfile.TemporaryDirectory() as td:
        res.save_to_json(td)
        json_files = [f for f in os.listdir(td) if f.endswith(".json")]
        if json_files:
            with open(os.path.join(td, json_files[0])) as f:
                data = json.load(f)
                with open(f"/tmp/debug_table_result_{idx}.json", "w") as out:
                    json.dump(data, out, indent=2)
                    print(f"Saved page {idx} to /tmp/debug_table_result_{idx}.json")
