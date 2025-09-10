import os
import csv
import time
from tqdm import tqdm

# Configuration
ROOT_DIRS = [
    'datasets/bradford25/audio17-18_annotations/firstOne',
    'datasets/bradford25/audio17-18_annotations/SecondOne',
    'datasets/bradford25/AudioMM',
    'datasets/bradford25/AudioTO'
]
TARGET_FILES = ['annotation.txt', 'Annotation.txt']
OUTPUT_CSV = 'extracted_annotations.csv'
CSV_HEADERS = [
    'file_path', 'operator', 'Time_start', 'Time_end', 
    'location', 'description', 'density'
]

def print_summary(total_files, processed_files, errors, elapsed_time):
    """Print processing summary with statistics"""
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files found:      {total_files}")
    print(f"Successfully processed: {processed_files - len(errors)}")
    print(f"Errors encountered:     {len(errors)}")
    print(f"Total time:            {elapsed_time:.2f} seconds")
    
    if errors:
        print("\nError details (first 5):")
        for error in errors[:5]:
            print(f"- {error}")
        if len(errors) > 5:
            print(f"- ... and {len(errors)-5} more errors")
    print("="*50 + "\n")

def clean_field(value):
    """Remove any quotes and extra spaces from fields"""
    return value.replace("'", "").replace('"', '').strip()

def parse_annotation_content(content):
    """Parse the specific format with proper location/description separation"""
    try:
        parts = content.strip().split()
        if len(parts) < 6:
            raise ValueError("Not enough components in the line")
        
        # Operator is always first
        data = {
            'operator': clean_field(parts[0]),
            'Time_start': clean_field(parts[1]),
            'Time_end': clean_field(parts[2]),
            'density': clean_field(parts[-1])  # Density is always last
        }
        
        # Find where the quoted descriptions begin
        desc_start = next((i for i, x in enumerate(parts) if x.startswith("'")), len(parts))
        
        # Location is everything between time_end and descriptions
        data['location'] = clean_field(' '.join(parts[3:desc_start]))
        
        # Description is all quoted terms combined (with quotes removed)
        desc_parts = [clean_field(p) for p in parts[desc_start:-1]]
        data['description'] = ' '.join(desc_parts)
        
        return data, None
        
    except Exception as e:
        return None, f"Content parsing error: {str(e)}"

def extract_file_content(filepath):
    """Extract and parse content from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        if not content:
            return None, f"{filepath}: Empty file"
            
        # Parse the content
        data, error = parse_annotation_content(content)
        if error:
            return None, f"{filepath}: {error}"
            
        # Add file path to the data
        data['file_path'] = filepath
        return data, None
        
    except Exception as e:
        return None, f"{filepath}: {str(e)}"

def find_annotation_files():
    """Find all annotation files in the specified directories"""
    annotation_files = []
    print("Searching for annotation files...")
    
    for root_dir in tqdm(ROOT_DIRS, desc="Directories scanned"):
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename in TARGET_FILES:
                    full_path = os.path.join(dirpath, filename)
                    annotation_files.append(full_path)
    
    return annotation_files

def main():
    start_time = time.time()
    errors = []
    
    # Phase 1: File discovery
    annotation_files = find_annotation_files()
    if not annotation_files:
        print("Error: No annotation files found in the specified paths.")
        return
    
    print(f"\nFound {len(annotation_files)} files to process\n")
    
    # Phase 2: Content extraction
    all_data = []
    processed_files = 0
    
    with tqdm(annotation_files, desc="Processing files") as progress_bar:
        for filepath in progress_bar:
            data, error = extract_file_content(filepath)
            if error:
                errors.append(error)
            else:
                all_data.append(data)
            processed_files += 1
    
    # Phase 3: Save results - ensure cleaned data is written
    if all_data:
        try:
            with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
                writer.writeheader()
                for row in all_data:
                    # Ensure all fields are properly cleaned before writing
                    cleaned_row = {
                        'file_path': row['file_path'],
                        'operator': clean_field(row['operator']),
                        'Time_start': clean_field(row['Time_start']),
                        'Time_end': clean_field(row['Time_end']),
                        'location': clean_field(row['location']),
                        'description': clean_field(row['description']),
                        'density': clean_field(row['density'])
                    }
                    writer.writerow(cleaned_row)
        except Exception as e:
            errors.append(f"Failed to write CSV: {str(e)}")
    
    # Reporting
    elapsed_time = time.time() - start_time
    print_summary(len(annotation_files), processed_files, errors, elapsed_time)
    
    if all_data:
        print(f"Data saved to {OUTPUT_CSV}")
        print(f"First few entries:")
        for i, entry in enumerate(all_data[:3]):
            print(f"\nEntry {i+1}:")
            for key, value in entry.items():
                print(f"{key:15}: {clean_field(value)}")
    else:
        print("Warning: No valid data was extracted")

if __name__ == '__main__':
    # Check for required packages
    try:
        from tqdm import tqdm
    except ImportError:
        print("Error: Required package 'tqdm' not installed.")
        print("Please install it with: pip install tqdm")
        exit(1)
    
    main()