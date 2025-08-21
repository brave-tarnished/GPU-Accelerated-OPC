import gdspy
import sqlite3
import numpy as np

def sqlite_to_gds(db_name, output_gds, layer_num=1, datatype=0):
    """
    Converts polygon data from an SQLite database to a GDS file.
    
    Args:
        db_name (str): SQLite database filename
        output_gds (str): Output GDS filename
        layer_num (int): Layer number to use for polygons (defaults to 1)
        datatype (int): Datatype to use for polygons (defaults to 0)
    """
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        
        # Create a new GDS library and cell
        lib = gdspy.GdsLibrary()
        cell = lib.new_cell('TOP')
        
        # Query to get distinct polygon IDs
        cursor.execute("SELECT DISTINCT polygon_id FROM processed_polygons ORDER BY polygon_id")
        polygon_ids = cursor.fetchall()
        
        total_polygons = 0
        
        # Process each polygon
        for (polygon_id,) in polygon_ids:
            # Get all points for this polygon
            cursor.execute("SELECT x, y FROM processed_polygons WHERE polygon_id = ? ORDER BY id", (polygon_id,))
            points = cursor.fetchall()
            if len(points) >= 3:  # Minimum 3 points required for a valid polygon
                # Convert points to numpy array
                polygon_points = np.array(points)
                
                # Create a polygon and add it to the cell
                polygon = gdspy.Polygon(polygon_points, layer=layer_num, datatype=datatype)
                cell.add(polygon)
                
                total_polygons += 1
        
        # Write the GDS file
        lib.write_gds(output_gds)
        
        # Close the database connection
        conn.close()
        
        print(f"Successfully converted SQLite data to GDS: {output_gds}")
        print(f"Total polygons processed: {total_polygons}")
    
    except Exception as e:
        print(f"Error during SQLite to GDS conversion: {str(e)}")

if __name__ == "__main__":
    # INPUT_DB = "DB Files/Divided/small_layout_block_0_0.db"
    # OUTPUT_GDS = "GDS Files/Divided/multiple_polygon_modified_0_0.gds"
    INPUT_DB = "DB Files/Divided/small_layout_block_0_0.db"
    OUTPUT_GDS = "GDS Files/Divided/small_layout_block_0_0.gds"    
    LAYER_NUMBER = 1
    
    sqlite_to_gds(INPUT_DB, OUTPUT_GDS, LAYER_NUMBER)