"""
Extract all tables from OCR result to CSV files
"""
import json
import os
from typing import List, Dict
import csv
from io import StringIO


def extract_table_to_csv_simple(table_id: str, blocks: List[Dict],
                                 x_min: float, x_max: float,
                                 y_min: float, y_max: float,
                                 img_width: int, img_height: int) -> str:
    """
    Extract table blocks and convert to CSV using simple row/column detection.
    """

    # Find all blocks within table bounds
    table_blocks = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x = sum(v['x'] for v in vertices) / 4
        block_y = sum(v['y'] for v in vertices) / 4

        if x_min <= block_x <= x_max and y_min <= block_y <= y_max:
            table_blocks.append({
                'text': block['text'],
                'x': block_x,
                'y': block_y,
                'x_min': min(v['x'] for v in vertices),
                'x_max': max(v['x'] for v in vertices),
                'y_min': min(v['y'] for v in vertices),
                'y_max': max(v['y'] for v in vertices),
            })

    if not table_blocks:
        return ""

    # Sort by Y position (top to bottom)
    table_blocks.sort(key=lambda b: b['y'])

    # Group into rows
    rows = []
    current_row = [table_blocks[0]]
    y_threshold = 0.008  # 0.8% of image height

    for block in table_blocks[1:]:
        current_row_y = sum(b['y'] for b in current_row) / len(current_row)

        if abs(block['y'] - current_row_y) < y_threshold:
            current_row.append(block)
        else:
            rows.append(current_row)
            current_row = [block]

    if current_row:
        rows.append(current_row)

    # Sort blocks within each row by X position (left to right)
    for row in rows:
        row.sort(key=lambda b: b['x'])

    # Find column positions by analyzing X coordinates across all rows
    all_x = sorted(set(b['x'] for row in rows for b in row))

    # Cluster X positions into columns
    columns = []
    if all_x:
        columns = [all_x[0]]
        x_cluster_threshold = 0.03  # 3% of image width

        for x in all_x[1:]:
            if x - columns[-1] > x_cluster_threshold:
                columns.append(x)

    # Build grid
    grid = []
    for row in rows:
        grid_row = [''] * len(columns)

        for block in row:
            # Find closest column
            min_dist = float('inf')
            col_idx = 0

            for i, col_x in enumerate(columns):
                dist = abs(block['x'] - col_x)
                if dist < min_dist:
                    min_dist = dist
                    col_idx = i

            # Add to grid (concatenate if cell already has content)
            if grid_row[col_idx]:
                grid_row[col_idx] += ' ' + block['text']
            else:
                grid_row[col_idx] = block['text']

        grid.append(grid_row)

    # Generate CSV
    output = StringIO()
    csv_writer = csv.writer(output)

    for row in grid:
        csv_writer.writerow(row)

    return output.getvalue()


def find_equipment_table(data: Dict) -> Dict:
    """
    Find equipment specification table in the bottom section.
    """
    page = data['pages'][0]
    blocks = page['blocks']

    # Look for equipment table keywords in bottom 20% of page
    equipment_keywords = ['EQUIPMENT', 'SIZE', 'DESIGN', 'OPERATING', 'TEMPERATURE', 'PRESSURE']

    bottom_blocks = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        y = sum(v['y'] for v in vertices) / 4

        if y > 0.85:  # Bottom 15%
            text_upper = block['text'].upper()
            if any(kw in text_upper for kw in equipment_keywords):
                bottom_blocks.append(block)

    if bottom_blocks:
        # Find bounds of equipment table
        all_vertices = []
        for block in bottom_blocks:
            all_vertices.extend(block['boundingPoly']['normalizedVertices'])

        x_min = min(v['x'] for v in all_vertices) - 0.05
        x_max = max(v['x'] for v in all_vertices) + 0.05
        y_min = min(v['y'] for v in all_vertices) - 0.02
        y_max = max(v['y'] for v in all_vertices) + 0.02

        return {
            'id': 'equipment-table',
            'x_min': max(0, x_min),
            'x_max': min(1, x_max),
            'y_min': max(0, y_min),
            'y_max': min(1, y_max)
        }

    return None


if __name__ == '__main__':
    # Load data
    with open('new_result.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']
    img_width = page['width']
    img_height = page['height']

    print('=== Extracting Tables from New Image ===\n')

    # Method 1: Extract from detected heuristic tables
    if 'tables' in page and page['tables']:
        print(f'Found {len(page["tables"])} heuristic table(s)\n')

        # If there's one big table covering everything, it's not useful
        # Let's look for specific tables instead

    # Method 2: Find equipment specification table
    equipment_table = find_equipment_table(data)

    if equipment_table:
        print(f'=== Equipment Specification Table ===')
        print(f'Position: x=[{equipment_table["x_min"]:.3f}, {equipment_table["x_max"]:.3f}], '
              f'y=[{equipment_table["y_min"]:.3f}, {equipment_table["y_max"]:.3f}]')

        x_min_px = int(equipment_table['x_min'] * img_width)
        x_max_px = int(equipment_table['x_max'] * img_width)
        y_min_px = int(equipment_table['y_min'] * img_height)
        y_max_px = int(equipment_table['y_max'] * img_height)

        print(f'Pixels: [{x_min_px}, {y_min_px}] to [{x_max_px}, {y_max_px}]')
        print(f'Size: {x_max_px - x_min_px} x {y_max_px - y_min_px} px\n')

        # Extract to CSV
        csv_content = extract_table_to_csv_simple(
            equipment_table['id'],
            blocks,
            equipment_table['x_min'],
            equipment_table['x_max'],
            equipment_table['y_min'],
            equipment_table['y_max'],
            img_width,
            img_height
        )

        if csv_content:
            print('CSV Content:')
            print('-' * 100)
            print(csv_content)
            print('-' * 100)

            filename = 'equipment_specification_table.csv'
            with open(filename, 'w', encoding='utf-8', newline='') as f:
                f.write(csv_content)
            print(f'\nSaved to: {filename}')
        else:
            print('No content extracted')
    else:
        print('No equipment table found')

    # Method 3: Try to find any other structured tables in specific regions
    # (title block, notes, legends, etc.)
    print('\n=== Searching for Other Tables ===')

    # Title block (usually top-left corner)
    title_block = {
        'id': 'title-block',
        'x_min': 0,
        'x_max': 0.3,
        'y_min': 0,
        'y_max': 0.25
    }

    csv_content = extract_table_to_csv_simple(
        title_block['id'],
        blocks,
        title_block['x_min'],
        title_block['x_max'],
        title_block['y_min'],
        title_block['y_max'],
        img_width,
        img_height
    )

    if csv_content:
        print('\n=== Title Block ===')
        filename = 'title_block.csv'
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        print(f'Saved to: {filename}')
        print(f'Rows: {len(csv_content.strip().split(chr(10)))}')
