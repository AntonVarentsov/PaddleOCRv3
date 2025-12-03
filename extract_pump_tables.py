"""
Extract pump specification tables (P-T6201A and P-T6201B) to clean CSV
"""
import json
from typing import List, Dict, Tuple
import csv


def extract_pump_tables_csv(blocks: List[Dict], img_width: int, img_height: int) -> Tuple[str, str]:
    """
    Extract two pump tables side by side and return as separate CSVs.

    Returns:
        Tuple of (pump_a_csv, pump_b_csv)
    """

    # Find the two pump name blocks to define table regions
    pump_a_block = None
    pump_b_block = None

    for block in blocks:
        text = block['text'].upper()
        if 'P-T6201A' in text and 'CRUDE OIL PUMP' in text:
            pump_a_block = block
        elif 'P-T6201B' in text and 'CRUDE OIL PUMP' in text:
            pump_b_block = block

    if not pump_a_block or not pump_b_block:
        return "", ""

    # Get pump positions
    pump_a_vertices = pump_a_block['boundingPoly']['normalizedVertices']
    pump_b_vertices = pump_b_block['boundingPoly']['normalizedVertices']

    pump_a_x = sum(v['x'] for v in pump_a_vertices) / 4
    pump_b_x = sum(v['x'] for v in pump_b_vertices) / 4

    # Define column boundaries
    # Column divider is midpoint between two pumps
    column_divider = (pump_a_x + pump_b_x) / 2

    # Find the left edge of Pump A data column (where values start, not labels)
    # Look for value blocks near pump A
    pump_a_value_x = []
    pump_b_value_x = []

    for block in blocks:
        text = block['text'].upper()
        # Look for value blocks (not labels)
        if any(val in text for val in ['VERTICAL', 'CRUDE OIL', '600', '130', '678', '33', 'ELECTRIC', '52']):
            vertices = block['boundingPoly']['normalizedVertices']
            block_x = sum(v['x'] for v in vertices) / 4
            block_y = sum(v['y'] for v in vertices) / 4

            # Check if near pumps vertically
            pump_y = sum(v['y'] for v in pump_a_vertices) / 4
            if abs(block_y - pump_y) < 0.1:  # Within 10% vertically
                if block_x < column_divider:
                    pump_a_value_x.append(block_x)
                else:
                    pump_b_value_x.append(block_x)

    # Define X boundaries for each pump column
    if pump_a_value_x:
        pump_a_x_min = min(pump_a_value_x) - 0.02
        pump_a_x_max = max(pump_a_value_x) + 0.02
    else:
        pump_a_x_min = pump_a_x - 0.1
        pump_a_x_max = column_divider

    if pump_b_value_x:
        pump_b_x_min = min(pump_b_value_x) - 0.02
        pump_b_x_max = max(pump_b_value_x) + 0.02
    else:
        pump_b_x_min = column_divider
        pump_b_x_max = pump_b_x + 0.1

    # Define table rows based on expected fields
    expected_rows = [
        'TYPE',
        'MEDIUM',
        'DESIGN PRESS',
        'DESIGN TEMP',
        'DESIGN CAPACITY',
        'DIFF. HEAD',
        'DRIVER',
        'HYDR. POWER'
    ]

    # Find Y range for the table (from TYPE to HYDR. POWER)
    y_min, y_max = 1.0, 0.0
    for block in blocks:
        text = block['text'].upper()
        if any(field in text for field in expected_rows):
            vertices = block['boundingPoly']['normalizedVertices']
            block_y = sum(v['y'] for v in vertices) / 4
            y_min = min(y_min, block_y - 0.01)
            y_max = max(y_max, block_y + 0.01)

    # Collect blocks for each table
    pump_a_data = {row: [] for row in expected_rows}
    pump_b_data = {row: [] for row in expected_rows}

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x = sum(v['x'] for v in vertices) / 4
        block_y = sum(v['y'] for v in vertices) / 4

        # Check if block is in table Y range
        if not (y_min <= block_y <= y_max):
            continue

        text = block['text']
        text_upper = text.upper()

        # Determine which row this belongs to
        matched_row = None
        for row_name in expected_rows:
            # Check if this block is a row label
            if row_name in text_upper:
                matched_row = row_name
                break

        # If this is a value block (not a label), find the closest row
        if matched_row is None:
            # Find closest row by Y position
            min_y_dist = float('inf')
            for row_name in expected_rows:
                # Find Y position of this row's label
                for other_block in blocks:
                    other_text = other_block['text'].upper()
                    if row_name in other_text:
                        other_vertices = other_block['boundingPoly']['normalizedVertices']
                        other_y = sum(v['y'] for v in other_vertices) / 4
                        y_dist = abs(block_y - other_y)
                        if y_dist < min_y_dist and y_dist < 0.015:  # Within 1.5% of image height
                            min_y_dist = y_dist
                            matched_row = row_name
                        break

        if matched_row:
            # Filter out label blocks
            is_label = (matched_row in text_upper and len(text) <= len(matched_row) + 5)
            if is_label:
                continue

            # Determine which pump column based on X boundaries
            if pump_a_x_min <= block_x <= pump_a_x_max:
                pump_a_data[matched_row].append(text)
            elif pump_b_x_min <= block_x <= pump_b_x_max:
                pump_b_data[matched_row].append(text)

    # Build CSV for each pump
    def build_csv(pump_name: str, data: Dict[str, List[str]]) -> str:
        output_lines = []
        output_lines.append(f'Parameter,Value')
        output_lines.append(f'Pump,{pump_name}')

        for row_name in expected_rows:
            values = data[row_name]
            if values:
                # Combine multiple values with space
                value_str = ' '.join(values)
            else:
                value_str = ''
            output_lines.append(f'{row_name},{value_str}')

        return '\n'.join(output_lines)

    pump_a_csv = build_csv('P-T6201A (100%) CRUDE OIL PUMP', pump_a_data)
    pump_b_csv = build_csv('P-T6201B (100%) CRUDE OIL PUMP', pump_b_data)

    return pump_a_csv, pump_b_csv


if __name__ == '__main__':
    # Load data
    with open('result_final.json') as f:
        data = json.load(f)

    page = data['pages'][0]
    blocks = page['blocks']
    img_width = page['width']
    img_height = page['height']

    print('=== Extracting Pump Specification Tables ===\n')

    pump_a_csv, pump_b_csv = extract_pump_tables_csv(blocks, img_width, img_height)

    if pump_a_csv:
        print('=== P-T6201A (100%) CRUDE OIL PUMP ===')
        print(pump_a_csv)
        print()

        with open('pump_T6201A.csv', 'w', encoding='utf-8') as f:
            f.write(pump_a_csv)
        print('Saved to: pump_T6201A.csv\n')

    if pump_b_csv:
        print('=== P-T6201B (100%) CRUDE OIL PUMP ===')
        print(pump_b_csv)
        print()

        with open('pump_T6201B.csv', 'w', encoding='utf-8') as f:
            f.write(pump_b_csv)
        print('Saved to: pump_T6201B.csv\n')
