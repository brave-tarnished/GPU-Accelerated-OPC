import gdspy
import math
import numpy as np
import sqlite3
import os

def gds_to_sqlite(input_gds, db_name, layer_num=1):
    """
    Converts a GDS file to an SQLite database while preserving polygon separation information.
    Each polygon's points are stored with a unique polygon ID.
    
    Args:
        input_gds (str): Input GDS filename
        db_name (str): SQLite database filename
        layer_num (int): Layer number to process (defaults to 1)
    """
    try:
        # Load the GDSII file
        lib = gdspy.GdsLibrary(infile=input_gds)
        
        # Connect to the SQLite database (creates if it doesn't exist)
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create table for storing polygon points
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS polygons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                polygon_id INTEGER,
                x REAL,
                y REAL
            )
        """)
        
        polygon_id = 0  # Unique identifier for each polygon
        
        # Iterate through all top-level cells in the GDS file
        for cell in lib.top_level():
            polygons = cell.get_polygons(by_spec=True).get((layer_num, 0), [])
            
            for polygon in polygons:
                polygon_id += 1  # Increment polygon ID for each new polygon
                for point in polygon:
                    cursor.execute("INSERT INTO polygons (polygon_id, x, y) VALUES (?, ?, ?)", 
                                   (polygon_id, point[0], point[1]))
                
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        print(f"Successfully stored GDS data in SQLite: {db_name}")
        print(f"Total polygons processed: {polygon_id}")
        return polygon_id
    
    except Exception as e:
        print(f"Error during GDS to SQLite conversion: {str(e)}")
        return 0

def divide_gds_file(input_gds, output_prefix, divisor, layer_num=1):
    """
    Divides a GDS file into smaller blocks and converts each block to an SQLite database.
    
    Args:
        input_gds (str): Input GDS filename
        output_prefix (str): Prefix for output filenames
        divisor (int): Number of blocks to divide the layout into
        layer_num (int): Layer number to process for DB conversion (defaults to 1)
    """
    # Load the input GDS file
    gdsii = gdspy.GdsLibrary()
    gdsii.read_gds(input_gds)
    
    # Get the top cell
    top_cell = gdsii.top_level()[0]
    
    # Calculate the bounding box of the entire layout
    bbox = top_cell.get_bounding_box()
    if bbox is None:
        print("Warning: Empty layout or unable to determine bounding box.")
        return
        
    total_width = bbox[1][0] - bbox[0][0]
    total_height = bbox[1][1] - bbox[0][1]
    
    print(f"Layout dimensions: {total_width} x {total_height}")
    
    # Calculate the size of each block
    block_width = total_width / math.sqrt(divisor)
    block_height = total_height / math.sqrt(divisor)
    
    # Create output directories if they don't exist
    gds_output_dir = os.path.dirname(os.path.join("GDS Files/Divided", f"{output_prefix}_block_0_0.gds"))
    db_output_dir = os.path.dirname(os.path.join("DB Files/Divided", f"{output_prefix}_block_0_0.db"))
    
    os.makedirs(gds_output_dir, exist_ok=True)
    os.makedirs(db_output_dir, exist_ok=True)
    
    # Divide the layout into blocks
    blocks_created = 0
    total_polygons = 0
    
    for i in range(int(math.sqrt(divisor))):
        for j in range(int(math.sqrt(divisor))):
            # Calculate the bounding box for this block
            x_min = bbox[0][0] + i * block_width
            y_min = bbox[0][1] + j * block_height
            x_max = x_min + block_width
            y_max = y_min + block_height
            
            print(f"Processing block ({i},{j}) with bounds: ({x_min},{y_min}) to ({x_max},{y_max})")
            
            # Create a new cell for this block
            block_cell = gdspy.Cell(f"BLOCK_{i}_{j}")
            
            # Get all polygons by specification
            polygons_by_spec = top_cell.get_polygons(by_spec=True)
            
            # Flag to check if any polygons were added to this block
            polygons_added = False
            
            # Iterate over the polygons
            for (layer, datatype), polys in polygons_by_spec.items():
                for poly_points in polys:
                    # Create a polygon object
                    poly = gdspy.Polygon(poly_points, layer=layer, datatype=datatype)
                    
                    # Get the polygon's bounding box
                    poly_bbox = poly.get_bounding_box()
                    
                    # Simple bounding box overlap check
                    if (poly_bbox[0][0] <= x_max and poly_bbox[1][0] >= x_min and
                        poly_bbox[0][1] <= y_max and poly_bbox[1][1] >= y_min):
                        # There's a potential overlap, do a more precise check
                        # Create a rectangle representing the current block
                        block_rect = gdspy.Rectangle((x_min, y_min), (x_max, y_max), layer=layer, datatype=datatype)
                        
                        # Check for intersection using boolean operation
                        try:
                            intersection = gdspy.boolean(poly, block_rect, 'and')
                            # Check if the intersection result is not None and contains polygons
                            if intersection is not None and len(intersection.polygons) > 0:
                                block_cell.add(poly)
                                polygons_added = True
                        except Exception as e:
                            print(f"Warning: Boolean operation failed: {e}")
                            # Fall back to simple bounding box check
                            block_cell.add(poly)
                            polygons_added = True
            
            # Only save blocks that contain polygons
            if polygons_added:
                # Create a new library for this block
                block_lib = gdspy.GdsLibrary()
                block_lib.add(block_cell)
                
                # Save the block as a separate GDS file
                gds_output_file = f"GDS Files/Divided/{output_prefix}_block_{i}_{j}.gds"
                block_lib.write_gds(gds_output_file)
                print(f"Created block file: {gds_output_file}")
                
                # Convert the GDS file to SQLite database
                db_output_file = f"DB Files/Divided/{output_prefix}_block_{i}_{j}.db"
                polygons_count = gds_to_sqlite(gds_output_file, db_output_file, layer_num)
                total_polygons += polygons_count
                
                blocks_created += 1
            else:
                print(f"Block ({i},{j}) is empty, skipping.")
    
    print(f"Divided the layout into {blocks_created} non-empty blocks out of {divisor} possible blocks.")
    print(f"Total polygons processed across all blocks: {total_polygons}")

# Usage example
if __name__ == "__main__":
    input_gds = "GDS Files/polygon.gds"
    output_prefix = "small_layout"
    divisor = 4
    layer_num = 1  # Layer number to process for DB conversion

    divide_gds_file(input_gds, output_prefix, divisor, layer_num)