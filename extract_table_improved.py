"""
Improved universal table extraction algorithm
Uses density-based clustering and structural analysis
"""
import json
import csv
from io import StringIO
from typing import List, Dict, Tuple
from collections import defaultdict


def calculate_block_density(blocks: List[Dict], x_min: float, x_max: float,
                            y_min: float, y_max: float,
                            grid_size: int = 20) -> float:
    """
    Calculate text density in a region by dividing it into a grid.
    Returns the percentage of grid cells that contain text.
    """
    grid = [[False] * grid_size for _ in range(grid_size)]

    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x_min = min(v['x'] for v in vertices)
        block_x_max = max(v['x'] for v in vertices)
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)

        # Check if block overlaps with region
        if block_x_max < x_min or block_x_min > x_max:
            continue
        if block_y_max < y_min or block_y_min > y_max:
            continue

        # Mark grid cells covered by this block
        x_start = int((max(block_x_min, x_min) - x_min) / (x_max - x_min) * grid_size)
        x_end = int((min(block_x_max, x_max) - x_min) / (x_max - x_min) * grid_size)
        y_start = int((max(block_y_min, y_min) - y_min) / (y_max - y_min) * grid_size)
        y_end = int((min(block_y_max, y_max) - y_min) / (y_max - y_min) * grid_size)

        for y in range(max(0, y_start), min(grid_size, x_end + 1)):
            for x in range(max(0, x_start), min(grid_size, y_end + 1)):
                grid[y][x] = True

    filled_cells = sum(sum(row) for row in grid)
    total_cells = grid_size * grid_size
    return filled_cells / total_cells if total_cells > 0 else 0


def find_table_boundaries(blocks: List[Dict],
                         initial_x_min: float, initial_x_max: float,
                         initial_y_min: float, initial_y_max: float,
                         density_threshold: float = 0.15) -> Tuple[float, float, float, float]:
    """
    Refine table boundaries by finding regions with high text density.
    """
    # Get all blocks in the initial region
    region_blocks = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x = sum(v['x'] for v in vertices) / 4
        block_y = sum(v['y'] for v in vertices) / 4

        if initial_x_min <= block_x <= initial_x_max and initial_y_min <= block_y <= initial_y_max:
            region_blocks.append(block)

    if not region_blocks:
        return initial_x_min, initial_x_max, initial_y_min, initial_y_max

    # Find tight bounds around all blocks
    all_x_min = min(min(v['x'] for v in b['boundingPoly']['normalizedVertices'])
                    for b in region_blocks)
    all_x_max = max(max(v['x'] for v in b['boundingPoly']['normalizedVertices'])
                    for b in region_blocks)
    all_y_min = min(min(v['y'] for v in b['boundingPoly']['normalizedVertices'])
                    for b in region_blocks)
    all_y_max = max(max(v['y'] for v in b['boundingPoly']['normalizedVertices'])
                    for b in region_blocks)

    return all_x_min, all_x_max, all_y_min, all_y_max


def detect_horizontal_separators(blocks: List[Dict],
                                 x_min: float, x_max: float,
                                 y_min: float, y_max: float,
                                 min_gap: float = 0.01) -> List[float]:
    """
    Detect horizontal gaps that likely represent row separators.
    Returns list of Y coordinates where gaps exist.
    """
    # Get Y coordinates of all blocks
    y_positions = []
    for block in blocks:
        vertices = block['boundingPoly']['normalizedVertices']
        block_x = sum(v['x'] for v in vertices) / 4
        block_y_min = min(v['y'] for v in vertices)
        block_y_max = max(v['y'] for v in vertices)

        if x_min <= block_x <= x_max and y_min <= block_y_min <= y_max:
            y_positions.append((block_y_min, block_y_max, 'block'))

    if not y_positions:
        return []

    # Sort by Y position
    y_positions.sort()

    # Find gaps between consecutive blocks
    separators = []
    for i in range(len(y_positions) - 1):
        current_bottom = y_positions[i][1]
        next_top = y_positions[i + 1][0]
        gap = next_top - current_bottom

        if gap >= min_gap:
            # Gap found - separator is at midpoint
            separators.append((current_bottom + next_top) / 2)

    return separators


def group_blocks_into_rows(blocks: List[Dict],
                           x_min: float, x_max: float,
                           y_min: float, y_max: float,
                           separators: List[float]) -> List[List[Dict]]:
    """
    Group blocks into rows using detected separators.
    """
    # Filter blocks in table region
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
        return []

    # If no separators, use simple Y-based grouping
    if not separators:
        table_blocks.sort(key=lambda b: b['y'])
        rows = []
        current_row = [table_blocks[0]]
        y_threshold = 0.008

        for block in table_blocks[1:]:
            current_row_y = sum(b['y'] for b in current_row) / len(current_row)
            if abs(block['y'] - current_row_y) < y_threshold:
                current_row.append(block)
            else:
                rows.append(current_row)
                current_row = [block]

        if current_row:
            rows.append(current_row)

        return rows

    # Group blocks by separator regions
    separator_positions = [y_min] + separators + [y_max]
    rows = [[] for _ in range(len(separator_positions) - 1)]

    for block in table_blocks:
        # Find which row this block belongs to
        block_y = block['y']
        for i in range(len(separator_positions) - 1):
            if separator_positions[i] <= block_y < separator_positions[i + 1]:
                rows[i].append(block)
                break

    # Remove empty rows and sort blocks within each row by X
    rows = [row for row in rows if row]
    for row in rows:
        row.sort(key=lambda b: b['x'])

    return rows


def detect_columns(rows: List[List[Dict]],
                   x_cluster_threshold: float = 0.03) -> List[float]:
    """
    Detect column positions by analyzing X coordinates across all rows.
    """
    # Collect all X positions
    all_x = []
    for row in rows:
        for block in row:
            all_x.append(block['x_min'])  # Use left edge for better alignment

    if not all_x:
        return []

    all_x = sorted(set(all_x))

    # Cluster X positions
    columns = [all_x[0]]
    for x in all_x[1:]:
        if x - columns[-1] > x_cluster_threshold:
            columns.append(x)

    return columns


def build_table_grid(rows: List[List[Dict]],
                     columns: List[float]) -> List[List[str]]:
    """
    Build a grid structure from rows and columns.
    """
    if not columns:
        # Fallback: each block is its own column
        grid = []
        for row in rows:
            grid.append([block['text'] for block in row])
        return grid

    grid = []
    for row in rows:
        grid_row = [''] * len(columns)

        for block in row:
            # Find closest column
            min_dist = float('inf')
            col_idx = 0

            for i, col_x in enumerate(columns):
                dist = abs(block['x_min'] - col_x)
                if dist < min_dist:
                    min_dist = dist
                    col_idx = i

            # Add to grid (concatenate if cell already has content)
            if grid_row[col_idx]:
                grid_row[col_idx] += ' ' + block['text']
            else:
                grid_row[col_idx] = block['text']

        grid.append(grid_row)

    return grid


def extract_table_universal(blocks: List[Dict],
                            x_min: float, x_max: float,
                            y_min: float, y_max: float,
                            img_width: int, img_height: int) -> str:
    """
    Universal table extraction algorithm.
    """
    # Step 1: Refine boundaries
    x_min, x_max, y_min, y_max = find_table_boundaries(
        blocks, x_min, x_max, y_min, y_max
    )

    # Step 2: Detect row separators
    separators = detect_horizontal_separators(blocks, x_min, x_max, y_min, y_max)

    print(f'Found {len(separators)} horizontal separators')

    # Step 3: Group into rows
    rows = group_blocks_into_rows(blocks, x_min, x_max, y_min, y_max, separators)

    if not rows:
        return ""

    print(f'Grouped into {len(rows)} rows')

    # Step 4: Detect columns
    columns = detect_columns(rows)

    print(f'Detected {len(columns)} columns')

    # Step 5: Build grid
    grid = build_table_grid(rows, columns)

    # Step 6: Generate CSV
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

    print('=== Universal Table Extraction ===\n')

    # Test on equipment table (upper left area based on screenshot)
    equipment_table = {
        'id': 'equipment-table-top',
        'x_min': 0.02,  # Left edge
        'x_max': 0.65,  # Extends to middle-right
        'y_min': 0.02,  # Top edge
        'y_max': 0.20   # About 20% down
    }

    print(f'=== Equipment Table (Top-Left) ===')
    print(f'Initial bounds: x=[{equipment_table["x_min"]:.3f}, {equipment_table["x_max"]:.3f}], '
          f'y=[{equipment_table["y_min"]:.3f}, {equipment_table["y_max"]:.3f}]\n')

    csv_content = extract_table_universal(
        blocks,
        equipment_table['x_min'],
        equipment_table['x_max'],
        equipment_table['y_min'],
        equipment_table['y_max'],
        img_width,
        img_height
    )

    if csv_content:
        print('\nCSV Content:')
        print('-' * 100)
        print(csv_content)
        print('-' * 100)

        filename = 'equipment_table_improved.csv'
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            f.write(csv_content)
        print(f'\nSaved to: {filename}')
    else:
        print('No content extracted')
