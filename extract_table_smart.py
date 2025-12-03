"""
Smart table extraction using key anchor points
"""
import json
import csv
from io import StringIO
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import re


def find_table_anchors(blocks: List[Dict]) -> Optional[Dict]:
    """
    Find key anchor blocks that define the table structure.
    Returns dict with column headers and their positions.
    """
    # Look for equipment IDs (column headers)
    equipment_pattern = r'^[VP]-T\d+'

    anchors = {
        'columns': [],  # List of (equipment_id, x_center, y_center)
        'y_min': None,
        'y_max': None,
    }

    for block in blocks:
        text = block['text'].strip()
        vertices = block['boundingPoly']['normalizedVertices']
        x_center = sum(v['x'] for v in vertices) / 4
        y_center = sum(v['y'] for v in vertices) / 4
        y_min = min(v['y'] for v in vertices)
        y_max = max(v['y'] for v in vertices)

        # Match equipment IDs
        if re.match(equipment_pattern, text):
            # Filter: only in bottom area (table area)
            if y_center > 0.90:
                anchors['columns'].append((text, x_center, y_center, y_min, y_max))

                if anchors['y_max'] is None or y_max > anchors['y_max']:
                    anchors['y_max'] = y_max

    if not anchors['columns']:
        return None

    # Sort columns by X position
    anchors['columns'].sort(key=lambda x: x[1])

    # Find Y_min: look for row header labels (rightmost column)
    row_labels = ['EQUIPMENT', 'DESCRIPTION', 'TYPE', 'DESIGN PRESSURE',
                  'DESIGN TEMPERATURE', 'OPERATING PRESSURE', 'OPERATING TEMPERATURE',
                  'DESIGN CAPACITY', 'POWER', 'DUTY']

    y_positions = []
    for block in blocks:
        text = block['text'].upper()
        if any(label in text for label in row_labels):
            vertices = block['boundingPoly']['normalizedVertices']
            x_max = max(v['x'] for v in vertices)
            y_min = min(v['y'] for v in vertices)

            # Row labels are typically right-aligned (x > 0.85)
            if x_max > 0.85:
                y_positions.append(y_min)

    if y_positions:
        anchors['y_min'] = min(y_positions) - 0.005  # Add small buffer

    return anchors


def find_row_labels(blocks: List[Dict], y_min: float, y_max: float,
                   x_threshold: float = 0.85) -> List[Tuple[str, float, float]]:
    """
    Find row labels (rightmost column).
    Returns list of (label_text, y_min, y_max) for each row.
    """
    labels = []

    # Keywords that identify actual row labels (not units or values)
    row_keywords = ['EQUIPMENT', 'ESCRIPTION', 'YPE', 'ESIGN', 'PERATING',
                   'CAPACITY', 'SIZE', 'PRESSURE', 'TEMPERATURE', 'OWER', 'DUTY',
                   'MATERIAL', 'CODE', 'VERTICAL', 'HORIZONTAL']

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        x_max = max(v['x'] for v in vertices)
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)
        y_center = sum(v['y'] for v in vertices) / 4

        # Filter: right-aligned and within Y bounds
        if x_max > x_threshold and y_min <= y_center <= y_max:
            text = block['text'].strip()

            # Only keep blocks with row keywords
            text_upper = text.upper()
            if any(kw in text_upper for kw in row_keywords):
                labels.append((text, block_y_min, block_y_max, y_center))

    # Sort by Y position
    labels.sort(key=lambda x: x[3])

    return labels


def extract_cell_value(blocks: List[Dict],
                       col_x_min: float, col_x_max: float,
                       row_y_min: float, row_y_max: float) -> str:
    """
    Extract text from a specific cell (column + row intersection).
    """
    cell_texts = []

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        x_center = sum(v['x'] for v in vertices) / 4
        y_center = sum(v['y'] for v in vertices) / 4

        # Check if block is within cell bounds
        if col_x_min <= x_center <= col_x_max and row_y_min <= y_center <= row_y_max:
            cell_texts.append(block['text'].strip())

    return ' '.join(cell_texts)


def extract_table_smart(blocks: List[Dict]) -> str:
    """
    Smart table extraction using anchor-based approach.
    """
    # Step 1: Find table anchors (column headers)
    anchors = find_table_anchors(blocks)

    if not anchors or not anchors['columns']:
        print('No table anchors found')
        return ""

    print(f'Found {len(anchors["columns"])} columns:')
    for col_id, x_c, y_c, _, _ in anchors['columns']:
        print(f'  - {col_id} (x={x_c:.3f}, y={y_c:.3f})')

    # Define table bounds
    if anchors['y_min'] is None:
        # Fallback: use column header positions
        anchors['y_min'] = min(y_min for _, _, _, y_min, _ in anchors['columns']) - 0.05

    y_min = anchors['y_min']
    y_max = anchors['y_max'] if anchors['y_max'] else 1.0

    print(f'\nTable Y bounds: [{y_min:.3f}, {y_max:.3f}]')

    # Step 2: Find row labels
    row_labels = find_row_labels(blocks, y_min, y_max)

    if not row_labels:
        print('No row labels found')
        return ""

    print(f'\nFound {len(row_labels)} rows:')
    for label, _, _, y_c in row_labels[:10]:  # Show first 10
        print(f'  - {label} (y={y_c:.3f})')

    # Step 3: Define column bounds
    columns_with_bounds = []
    for i, (col_id, x_c, y_c, col_y_min, col_y_max) in enumerate(anchors['columns']):
        if i == 0:
            # First column: left edge to midpoint with next
            x_min = x_c - 0.05
            if len(anchors['columns']) > 1:
                x_max = (x_c + anchors['columns'][i+1][1]) / 2
            else:
                x_max = x_c + 0.05
        elif i == len(anchors['columns']) - 1:
            # Last column: midpoint with previous to right edge
            x_min = (anchors['columns'][i-1][1] + x_c) / 2
            x_max = x_c + 0.05
        else:
            # Middle columns: midpoint with neighbors
            x_min = (anchors['columns'][i-1][1] + x_c) / 2
            x_max = (x_c + anchors['columns'][i+1][1]) / 2

        columns_with_bounds.append((col_id, x_min, x_max, x_c))

    # Step 4: Define row bounds
    rows_with_bounds = []
    for i, (label, label_y_min, label_y_max, y_c) in enumerate(row_labels):
        if i == 0:
            # First row: from table top to midpoint with next
            row_y_min = y_min
            if len(row_labels) > 1:
                row_y_max = (y_c + row_labels[i+1][3]) / 2
            else:
                row_y_max = y_max
        elif i == len(row_labels) - 1:
            # Last row: midpoint with previous to table bottom
            row_y_min = (row_labels[i-1][3] + y_c) / 2
            row_y_max = y_max
        else:
            # Middle rows: midpoint with neighbors
            row_y_min = (row_labels[i-1][3] + y_c) / 2
            row_y_max = (y_c + row_labels[i+1][3]) / 2

        rows_with_bounds.append((label, row_y_min, row_y_max))

    # Step 5: Extract cell values
    grid = []

    for row_label, row_y_min, row_y_max in rows_with_bounds:
        row_data = [row_label]  # First column is row label

        for col_id, col_x_min, col_x_max, col_x_c in columns_with_bounds:
            cell_value = extract_cell_value(blocks, col_x_min, col_x_max,
                                           row_y_min, row_y_max)
            row_data.append(cell_value)

        grid.append(row_data)

    # Step 6: Add header row
    header_row = [''] + [col_id for col_id, _, _, _ in columns_with_bounds]
    grid.insert(0, header_row)

    # Step 7: Merge duplicate rows (rows with empty first column or very close Y positions)
    merged_grid = []
    skip_next = set()

    for i, row in enumerate(grid):
        if i in skip_next:
            continue

        # Check if next row should be merged
        if i + 1 < len(grid):
            next_row = grid[i + 1]

            # If next row has empty first column, merge its data into current row
            if not next_row[0] or len(next_row[0].strip()) == 0:
                for j in range(1, len(row)):
                    if j < len(next_row) and next_row[j] and not row[j]:
                        row[j] = next_row[j]
                skip_next.add(i + 1)

        merged_grid.append(row)

    # Step 8: Generate CSV
    output = StringIO()
    csv_writer = csv.writer(output)

    for row in merged_grid:
        csv_writer.writerow(row)

    return output.getvalue()


if __name__ == '__main__':
    # Load data
    with open('new_result.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']

    print('=== Smart Table Extraction ===\n')

    csv_content = extract_table_smart(blocks)

    if csv_content:
        print('\n' + '=' * 100)
        print('CSV Content:')
        print('=' * 100)
        print(csv_content)
        print('=' * 100)

        filename = 'equipment_table_smart.csv'
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        print(f'\nSaved to: {filename}')

        # Display row count
        row_count = len(csv_content.strip().split('\n'))
        print(f'Total rows: {row_count}')
    else:
        print('No content extracted')
