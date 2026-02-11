#!/usr/bin/env python3
"""
Script to add 'from __future__ import division, print_function' to all Python files.

This is critical for Python 3 compatibility to ensure:
1. Division operator / performs true division (3/2 = 1.5)
2. Print is a function, not a statement
"""

import os
import sys
from pathlib import Path


def has_future_import(content: str) -> bool:
    """Check if file already has future import."""
    lines = content.split('\n')
    for line in lines[:20]:  # Check first 20 lines
        if 'from __future__ import' in line:
            return True
    return False


def add_future_import(file_path: Path) -> bool:
    """Add future import to a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Skip if already has future import
        if has_future_import(content):
            print(f"SKIP: {file_path} (already has future import)")
            return False

        lines = content.split('\n')

        # Find where to insert (after docstring and encoding)
        insert_pos = 0
        in_docstring = False
        docstring_char = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Handle encoding declarations
            if i == 0 and ('coding' in stripped or 'encoding' in stripped):
                insert_pos = i + 1
                continue

            # Handle docstrings
            if stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_char = stripped[:3]
                    if stripped.endswith(docstring_char) and len(stripped) > 6:
                        # Single-line docstring
                        insert_pos = i + 1
                        in_docstring = False
                elif stripped.endswith(docstring_char):
                    # End of multi-line docstring
                    insert_pos = i + 1
                    in_docstring = False
                continue

            # If we're past docstrings and find first import or code
            if not in_docstring and stripped and not stripped.startswith('#'):
                insert_pos = i
                break

        # Insert the future import
        future_line = 'from __future__ import division, print_function'

        # Add blank line before if needed
        if insert_pos < len(lines) and lines[insert_pos].strip():
            lines.insert(insert_pos, future_line)
            lines.insert(insert_pos + 1, '')
        else:
            lines.insert(insert_pos, future_line)

        # Write back
        new_content = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"UPDATED: {file_path}")
        return True

    except Exception as e:
        print(f"ERROR: {file_path}: {e}", file=sys.stderr)
        return False


def main():
    python_dir = Path('python/gps')

    if not python_dir.exists():
        print(f"ERROR: {python_dir} does not exist!", file=sys.stderr)
        sys.exit(1)

    # Find all .py files
    py_files = list(python_dir.rglob('*.py'))
    print(f"Found {len(py_files)} Python files")

    updated = 0
    skipped = 0
    errors = 0

    for py_file in sorted(py_files):
        result = add_future_import(py_file)
        if result is True:
            updated += 1
        elif result is False:
            skipped += 1
        else:
            errors += 1

    print(f"\nSummary:")
    print(f"  Updated: {updated}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")


if __name__ == '__main__':
    main()
