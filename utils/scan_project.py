import os

def scan_directory(directory, output_file=None):
    """
    Scans a directory and prints its structure.

    Args:
        directory (str): The path to the project directory.
        output_file (str): Path to save the output as a text file (optional).
    """
    result = []

    for root, dirs, files in os.walk(directory):
        # Get folder level for indentation
        level = root.replace(directory, '').count(os.sep)
        indent = '│   ' * level  # Corrected indent to avoid potential issues
        result.append(f"{indent}├── {os.path.basename(root)}/")

        # List files in the current directory
        subindent = '│   ' * (level + 1) # Corrected indent to avoid potential issues
        for file in files:
            result.append(f"{subindent}└── {file}")

    # Print or save to file
    result_str = '\n'.join(result)
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f: # Added encoding='utf-8'
            f.write(result_str)
        print(f"✅ Project structure saved to: {output_file}")
    else:
        print(result_str)

# Path to your project directory
project_dir = os.path.abspath(os.path.join(os.getcwd(), '.'))  # One level up from 'utils'

# Run the directory scan
scan_directory(project_dir, output_file='project_structure.txt')