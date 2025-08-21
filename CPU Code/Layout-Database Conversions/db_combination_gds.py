import sqlite3
import gdspy
import os
import numpy as np

# def are_polygons_identical(poly1, poly2, tolerance=1e-10):
#     """
#     Check if two polygons have identical vertices (within a small tolerance).
    
#     Args:
#         poly1, poly2: Numpy arrays of polygon vertices
#         tolerance: Floating point comparison tolerance
        
#     Returns:
#         Boolean indicating if polygons are identical
#     """
#     if len(poly1) != len(poly2):
#         return False
    
#     # Check if all points match (allowing for small numerical differences)
#     # return False
#     return np.allclose(poly1, poly2, atol=tolerance)

def are_polygons_identical(poly1, poly2, tolerance=1e-10):
    """Temporarily disabled to check if duplicate removal is the issue"""
    return False  # Always return False to keep all polygons

def db_to_gds_with_duplicate_removal(db_files, output_gds, layer_num=1, datatype=0):
    """
    Converts multiple SQLite database files to a single GDS file,
    removing exact duplicate polygons.
    """
    # List to store unique polygons
    unique_polygons = []
    
    # Track statistics
    total_polygons = 0
    duplicate_count = 0
    
    print(f"Starting conversion of {len(db_files)} DB files to a single GDS file...")
    
    # Process each DB file
    for db_file in db_files:
        try:
            print(f"Processing database: {db_file}")
            
            # Connect to the SQLite database
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get the maximum polygon_id
            cursor.execute("SELECT MAX(polygon_id) FROM processed_polygons")
            max_polygon_id = cursor.fetchone()[0]
            
            if max_polygon_id is None:
                print(f"No polygons found in {db_file}, skipping...")
                conn.close()
                continue
            
            print(f"Found {max_polygon_id} polygons in {db_file}")
            
            # Process each polygon
            for polygon_id in range(1, max_polygon_id + 1):
                # Get all points for this polygon
                cursor.execute("SELECT x, y FROM processed_polygons WHERE polygon_id = ? ORDER BY id", (polygon_id,))
                points = cursor.fetchall()
                
                if not points or len(points) < 3:  # Need at least 3 points for a valid polygon
                    continue
                
                # Convert to numpy array
                points_array = np.array(points)
                
                # Check if this polygon is a duplicate of any existing polygon
                is_duplicate = False
                for existing_poly in unique_polygons:
                    if are_polygons_identical(points_array, existing_poly):
                        is_duplicate = True
                        duplicate_count += 1
                        break
                
                if not is_duplicate:
                    unique_polygons.append(points_array)
                
                total_polygons += 1
                
                # Print progress for large databases
                if polygon_id % 1000 == 0:
                    print(f"Processed {polygon_id} polygons from {db_file}...")
            
            # Close the database connection
            conn.close()
            
        except Exception as e:
            print(f"Error processing {db_file}: {str(e)}")
    
    # Create a new GDS library and cell
    gds_lib = gdspy.GdsLibrary()
    top_cell = gdspy.Cell("TOP")
    
    # Add all unique polygons to the cell
    for poly_points in unique_polygons:
        polygon = gdspy.Polygon(poly_points, layer=layer_num, datatype=datatype)
        top_cell.add(polygon)
    
    # Add the top cell to the library
    gds_lib.add(top_cell)
    
    # Write the GDS file
    gds_lib.write_gds(output_gds)
    
    print(f"Successfully converted {len(db_files)} DB files to {output_gds}")
    print(f"Total polygons processed: {total_polygons}")
    print(f"Duplicate polygons removed: {duplicate_count}")
    print(f"Final polygon count in GDS: {len(unique_polygons)}")
    
    return len(unique_polygons)

def find_db_files(directory, prefix=""):
    """
    Find all SQLite database files in a directory with an optional prefix.
    """
    db_files = []
    for file in os.listdir(directory):
        if file.endswith(".db") and file.startswith(prefix):
            db_files.append(os.path.join(directory, file))
    return sorted(db_files)  # Sort to process in a consistent order

def combine_divided_dbs_to_gds(db_directory, output_gds, prefix=""):
    """
    Combines all divided DB files in a directory back into a single GDS file,
    removing duplicate polygons.
    """
    # Find all DB files in the directory
    db_files = find_db_files(db_directory, prefix)
    
    if not db_files:
        print(f"No DB files found in {db_directory} with prefix '{prefix}'")
        return 0
    
    print(f"Found {len(db_files)} DB files to combine")
    
    # Convert the DB files to a single GDS with duplicate removal
    return db_to_gds_with_duplicate_removal(db_files, output_gds)

if __name__ == "__main__":
    # Example usage
    db_directory = "DB Files/Divided"
    output_gds = "GDS Files/polygon_modified.gds"
    prefix = "small_layout"  # Optional prefix to filter DB files
    
    # Combine all divided DB files back into a single GDS
    combine_divided_dbs_to_gds(db_directory, output_gds, prefix)