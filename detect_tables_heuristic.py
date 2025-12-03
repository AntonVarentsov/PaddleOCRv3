"""
Heuristic table detection based on OCR text block geometry
"""
import json
from collections import defaultdict
from typing import List, Dict, Tuple

def detect_tables_from_text_blocks(blocks: List[Dict], img_width: int, img_height: int,
                                   threshold: float = 0.05) -> List[Dict]:
    """
    Detect table-like structures from OCR text blocks using geometric heuristics.

    Args:
        blocks: List of text blocks with boundingPoly
        img_width: Image width
        img_height: Image height
        threshold: Alignment threshold (fraction of image size)

    Returns:
        List of detected table regions
    """
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
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'x_center': x_center,
            'y_center': y_center,
            'width': x_max - x_min,
            'height': y_max - y_min
        })

    # Find aligned rows (blocks with similar y-coordinates)
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

            # Check if blocks are on the same row
            if abs(info['y_center'] - other['y_center']) < y_threshold:
                row.append(other)
                used_blocks.add(j)

        if len(row) >= 2:  # At least 2 blocks in a row
            rows.append(sorted(row, key=lambda x: x['x_center']))

    if len(rows) < 2:
        return []

    # Sort rows by y position
    rows = sorted(rows, key=lambda r: r[0]['y_center'])

    # Find table candidates: groups of consecutive rows with similar column structure
    tables = []
    current_table_rows = []

    for row in rows:
        if not current_table_rows:
            current_table_rows.append(row)
            continue

        # Check if this row is close to the previous one and has similar structure
        prev_row = current_table_rows[-1]
        y_gap = row[0]['y_center'] - prev_row[0]['y_center']

        # Rows should be close together (< 10% of image height)
        if y_gap < 0.1:
            # Check column alignment
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
                # Save current table if it has enough rows
                if len(current_table_rows) >= 3:
                    tables.append(current_table_rows)
                current_table_rows = [row]
        else:
            # Save current table if it has enough rows
            if len(current_table_rows) >= 3:
                tables.append(current_table_rows)
            current_table_rows = [row]

    # Don't forget the last table
    if len(current_table_rows) >= 3:
        tables.append(current_table_rows)

    # Create table regions
    table_regions = []
    for idx, table_rows in enumerate(tables):
        # Find bounding box
        all_x = []
        all_y = []
        for row in table_rows:
            for block_info in row:
                all_x.extend([block_info['x_min'], block_info['x_max']])
                all_y.extend([block_info['y_min'], block_info['y_max']])

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        # Add some padding
        padding = 0.01
        x_min = max(0, x_min - padding)
        x_max = min(1, x_max + padding)
        y_min = max(0, y_min - padding)
        y_max = min(1, y_max + padding)

        table_regions.append({
            'id': f'table-heuristic-{idx + 1}',
            'type': 'table',
            'confidence': 0.85,  # Heuristic confidence
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


if __name__ == '__main__':
    # Test with our data
    with open('result_with_layout.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']
    img_width = page['width']
    img_height = page['height']

    print(f'Analyzing {len(blocks)} text blocks...')
    tables = detect_tables_from_text_blocks(blocks, img_width, img_height, threshold=0.02)

    print(f'\nFound {len(tables)} table(s):')
    for table in tables:
        bbox = table['boundingPoly']['normalizedVertices']
        print(f"\n{table['id']}:")
        print(f"  Rows: {table['rows']}")
        print(f"  Position: x=[{bbox[0]['x']:.3f}, {bbox[1]['x']:.3f}], y=[{bbox[0]['y']:.3f}, {bbox[2]['y']:.3f}]")

        # Calculate pixel coordinates
        x_min_px = int(bbox[0]['x'] * img_width)
        x_max_px = int(bbox[1]['x'] * img_width)
        y_min_px = int(bbox[0]['y'] * img_height)
        y_max_px = int(bbox[2]['y'] * img_height)
        print(f"  Pixels: x=[{x_min_px}, {x_max_px}], y=[{y_min_px}, {y_max_px}]")
        print(f"  Size: {x_max_px - x_min_px} x {y_max_px - y_min_px} px")
