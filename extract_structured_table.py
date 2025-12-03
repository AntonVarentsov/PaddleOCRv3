"""
Extract structured table using row headers detection
"""
import json
import csv
from io import StringIO
from typing import List, Dict, Tuple
from collections import defaultdict


def find_row_headers(blocks: List[Dict],
                     y_min: float, y_max: float,
                     header_x_threshold: float = 0.90) -> List[Tuple[str, float, float]]:
    """
    Find row headers - typically text blocks aligned to the right edge of the table.
    Returns list of (header_text, y_min, y_max) tuples.
    """
    headers = []

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x_max = max(v['x'] for v in vertices)
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)
        block_y_center = sum(v['y'] for v in vertices) / 4

        # Check if block is in the right edge area and within Y bounds
        if block_x_max > header_x_threshold and y_min <= block_y_center <= y_max:
            # Clean up header text
            text = block['text'].strip()
            # Filter out revision marks and other non-header text
            if len(text) > 2 and not text.isdigit():
                headers.append((text, block_y_min, block_y_max, block_y_center))

    # Sort by Y position
    headers.sort(key=lambda x: x[3])

    return headers


def extract_row_data(blocks: List[Dict],
                    row_y_min: float, row_y_max: float,
                    x_min: float, x_max: float,
                    y_tolerance: float = 0.005) -> List[Dict]:
    """
    Extract all data blocks within a row's Y range.
    """
    row_blocks = []

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)
        block_y_center = sum(v['y'] for v in vertices) / 4
        block_x_min = min(v['x'] for v in vertices)
        block_x_max = max(v['x'] for v in vertices)
        block_x_center = sum(v['x'] for v in vertices) / 4

        # Check if block overlaps with row Y range
        overlaps_y = not (block_y_max < row_y_min - y_tolerance or
                         block_y_min > row_y_max + y_tolerance)

        # Check if block is within X bounds
        within_x = x_min <= block_x_center <= x_max

        if overlaps_y and within_x:
            row_blocks.append({
                'text': block['text'],
                'x_min': block_x_min,
                'x_max': block_x_max,
                'x_center': block_x_center,
                'y_min': block_y_min,
                'y_max': block_y_max,
                'y_center': block_y_center,
            })

    # Sort by X position
    row_blocks.sort(key=lambda b: b['x_center'])

    return row_blocks


def detect_column_structure(all_row_blocks: List[List[Dict]],
                            col_threshold: float = 0.03) -> List[float]:
    """
    Detect column positions by analyzing X coordinates across all rows.
    """
    # Collect all X positions (use left edge for better alignment)
    all_x = []
    for row_blocks in all_row_blocks:
        for block in row_blocks:
            all_x.append(block['x_min'])

    if not all_x:
        return []

    all_x = sorted(set(all_x))

    # Cluster X positions into columns
    columns = [all_x[0]]
    for x in all_x[1:]:
        if x - columns[-1] > col_threshold:
            columns.append(x)

    return columns


def assign_blocks_to_columns(row_blocks: List[Dict],
                             columns: List[float],
                             col_tolerance: float = 0.04) -> List[str]:
    """
    Assign blocks to columns and return row data.
    """
    if not columns:
        return [block['text'] for block in row_blocks]

    grid_row = [''] * len(columns)

    for block in row_blocks:
        # Find closest column
        min_dist = float('inf')
        col_idx = 0

        for i, col_x in enumerate(columns):
            dist = abs(block['x_min'] - col_x)
            if dist < min_dist:
                min_dist = dist
                col_idx = i

        # Only assign if within tolerance
        if min_dist < col_tolerance:
            # Concatenate if cell already has content
            if grid_row[col_idx]:
                grid_row[col_idx] += ' ' + block['text']
            else:
                grid_row[col_idx] = block['text']

    return grid_row


def extract_structured_table(blocks: List[Dict],
                            y_min: float, y_max: float,
                            x_min: float = 0.0, x_max: float = 0.95,
                            header_x_threshold: float = 0.90) -> str:
    """
    Extract table using row header detection.

    Args:
        blocks: List of text blocks from OCR
        y_min, y_max: Y bounds of table region
        x_min, x_max: X bounds for data columns (excluding header column)
        header_x_threshold: X threshold for detecting row headers (headers are right-aligned)
    """
    # Step 1: Find row headers
    headers = find_row_headers(blocks, y_min, y_max, header_x_threshold)

    if not headers:
        print('No row headers found')
        return ""

    print(f'Found {len(headers)} row headers:')
    for header_text, _, _, y_c in headers:
        print(f'  - {header_text} (y={y_c:.3f})')

    # Step 2: Extract data for each row
    all_row_blocks = []
    rows_data = []

    for i, (header_text, h_y_min, h_y_max, h_y_center) in enumerate(headers):
        # Define row bounds (from previous header to this header)
        if i == 0:
            row_y_min = y_min
        else:
            row_y_min = headers[i-1][3]  # Previous header's center

        if i == len(headers) - 1:
            row_y_max = y_max
        else:
            row_y_max = headers[i+1][3]  # Next header's center

        # Extract blocks for this row
        row_blocks = extract_row_data(blocks, row_y_min, row_y_max, x_min, x_max)

        if row_blocks:
            all_row_blocks.append(row_blocks)
            rows_data.append((header_text, row_blocks))

    print(f'\nExtracted data for {len(rows_data)} rows')

    # Step 3: Detect column structure
    columns = detect_column_structure(all_row_blocks)

    print(f'Detected {len(columns)} columns at X positions: {[f"{x:.3f}" for x in columns]}')

    # Step 4: Build grid
    grid = []

    for header_text, row_blocks in rows_data:
        grid_row = assign_blocks_to_columns(row_blocks, columns)
        # Prepend header as first column
        grid.append([header_text] + grid_row)

    # Step 5: Generate CSV
    output = StringIO()
    csv_writer = csv.writer(output)

    for row in grid:
        csv_writer.writerow(row)

    return output.getvalue()


if __name__ == '__main__':
    # Load data
    with open('new_result.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']
    img_width = page['width']
    img_height = page['height']

    print('=== Structured Table Extraction ===\n')

    # Equipment specification table is at the bottom
    # Based on analysis: y > 0.92, headers at x > 0.90
    print('=== Equipment Specification Table ===\n')

    csv_content = extract_structured_table(
        blocks,
        y_min=0.91,      # Start of table area
        y_max=1.00,      # Bottom of page
        x_min=0.60,      # Left edge of data columns
        x_max=0.92,      # Right edge of data columns (before header column)
        header_x_threshold=0.90  # Headers are right-aligned at x > 0.90
    )

    if csv_content:
        print('\nCSV Content:')
        print('-' * 100)
        print(csv_content)
        print('-' * 100)

        filename = 'equipment_table_structured.csv'
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        print(f'\nSaved to: {filename}')

        # Display row count
        row_count = len(csv_content.strip().split('\n'))
        print(f'Total rows: {row_count}')
    else:
        print('No content extracted')
