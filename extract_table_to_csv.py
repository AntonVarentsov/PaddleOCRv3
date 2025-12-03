"""
Extract table data to CSV format based on text block positions
"""
import json
from typing import List, Dict, Tuple
import csv
from io import StringIO


def extract_table_to_csv(table: Dict, blocks: List[Dict], img_width: int, img_height: int) -> str:
    """
    Extract table content to CSV format by analyzing text block positions.

    Algorithm:
    1. Find all text blocks within the table bounding box
    2. Group blocks into rows based on Y-coordinate alignment
    3. Within each row, sort blocks by X-coordinate (left to right)
    4. Determine column boundaries by analyzing X-positions across all rows
    5. Assign each block to appropriate row and column
    6. Generate CSV output
    """

    # Get table bounds
    bbox = table['boundingPoly']['normalizedVertices']
    table_x_min = bbox[0]['x']
    table_x_max = bbox[1]['x']
    table_y_min = bbox[0]['y']
    table_y_max = bbox[2]['y']

    # Step 1: Find all blocks within table
    table_blocks = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x_min = min(v['x'] for v in vertices)
        block_x_max = max(v['x'] for v in vertices)
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)

        block_x_center = (block_x_min + block_x_max) / 2
        block_y_center = (block_y_min + block_y_max) / 2

        # Check if block center is within table bounds
        if (table_x_min <= block_x_center <= table_x_max and
            table_y_min <= block_y_center <= table_y_max):
            table_blocks.append({
                'text': block['text'],
                'confidence': block['confidence'],
                'x_min': block_x_min,
                'x_max': block_x_max,
                'y_min': block_y_min,
                'y_max': block_y_max,
                'x_center': block_x_center,
                'y_center': block_y_center
            })

    if not table_blocks:
        return ""

    # Step 2: Group blocks into rows by Y-coordinate
    # Sort by Y position first
    table_blocks.sort(key=lambda b: b['y_center'])

    rows = []
    current_row = [table_blocks[0]]
    y_threshold = 0.01  # Blocks within 1% of image height are considered same row

    for block in table_blocks[1:]:
        # Check if this block belongs to current row
        current_row_y = sum(b['y_center'] for b in current_row) / len(current_row)

        if abs(block['y_center'] - current_row_y) < y_threshold:
            current_row.append(block)
        else:
            # Start new row
            rows.append(current_row)
            current_row = [block]

    # Don't forget last row
    if current_row:
        rows.append(current_row)

    # Step 3: Sort blocks within each row by X-coordinate
    for row in rows:
        row.sort(key=lambda b: b['x_center'])

    # Step 4: Determine column boundaries
    # Collect all X positions
    all_x_positions = []
    for row in rows:
        for block in row:
            all_x_positions.append(block['x_center'])

    all_x_positions.sort()

    # Find column boundaries by clustering X positions
    if not all_x_positions:
        return ""

    column_centers = [all_x_positions[0]]
    x_cluster_threshold = 0.03  # 3% of image width

    for x in all_x_positions[1:]:
        if x - column_centers[-1] > x_cluster_threshold:
            column_centers.append(x)

    # Step 5: Assign blocks to grid
    num_cols = len(column_centers)
    grid = [['' for _ in range(num_cols)] for _ in range(len(rows))]

    for row_idx, row in enumerate(rows):
        for block in row:
            # Find closest column
            col_idx = 0
            min_distance = abs(block['x_center'] - column_centers[0])

            for i, col_center in enumerate(column_centers[1:], 1):
                distance = abs(block['x_center'] - col_center)
                if distance < min_distance:
                    min_distance = distance
                    col_idx = i

            # Place text in grid (append if cell already has content)
            if grid[row_idx][col_idx]:
                grid[row_idx][col_idx] += ' ' + block['text']
            else:
                grid[row_idx][col_idx] = block['text']

    # Step 6: Generate CSV
    output = StringIO()
    csv_writer = csv.writer(output)

    for row in grid:
        csv_writer.writerow(row)

    return output.getvalue()


def find_pump_tables(data: Dict) -> List[Dict]:
    """
    Find tables containing pump specifications based on keywords.
    """
    page = data['pages'][0]

    if 'tables' not in page:
        return []

    pump_keywords = ['P-T6201A', 'P-T6201B', 'CRUDE OIL PUMP', 'TYPE', 'MEDIUM', 'DESIGN']
    pump_tables = []

    for table in page['tables']:
        # Get table bounds
        bbox = table['boundingPoly']['normalizedVertices']
        x_min, x_max = bbox[0]['x'], bbox[1]['x']
        y_min, y_max = bbox[0]['y'], bbox[2]['y']

        # Count how many pump-related keywords appear in this table
        keyword_count = 0
        for block in page['blocks']:
            vertices = block['boundingPoly']['normalizedVertices']
            block_x = (vertices[0]['x'] + vertices[1]['x']) / 2
            block_y = (vertices[0]['y'] + vertices[2]['y']) / 2

            if x_min <= block_x <= x_max and y_min <= block_y <= y_max:
                text_upper = block['text'].upper()
                if any(kw in text_upper for kw in pump_keywords):
                    keyword_count += 1

        if keyword_count >= 3:  # If table contains at least 3 pump keywords
            pump_tables.append(table)

    return pump_tables


if __name__ == '__main__':
    # Load data
    with open('result_final.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']
    img_width = page['width']
    img_height = page['height']

    print('=== Extracting Tables to CSV ===\n')

    # Find pump specification tables
    pump_tables = find_pump_tables(data)

    print(f'Found {len(pump_tables)} pump specification table(s)\n')

    for idx, table in enumerate(pump_tables, 1):
        print(f'=== Table {idx}: {table["id"]} ===')
        print(f'Rows: {table.get("rows", "N/A")}')

        bbox = table['boundingPoly']['normalizedVertices']
        x_min_px = int(bbox[0]['x'] * img_width)
        x_max_px = int(bbox[1]['x'] * img_width)
        y_min_px = int(bbox[0]['y'] * img_height)
        y_max_px = int(bbox[2]['y'] * img_height)
        print(f'Position: [{x_min_px}, {y_min_px}] to [{x_max_px}, {y_max_px}] px')
        print(f'Size: {x_max_px - x_min_px} x {y_max_px - y_min_px} px\n')

        # Extract to CSV
        csv_content = extract_table_to_csv(table, blocks, img_width, img_height)

        if csv_content:
            print('CSV Content:')
            print('-' * 80)
            print(csv_content)
            print('-' * 80)

            # Save to file
            filename = f'table_{table["id"]}.csv'
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_content)
            print(f'Saved to: {filename}\n')
        else:
            print('No content extracted\n')
