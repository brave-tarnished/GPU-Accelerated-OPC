import gdspy
import sqlite3

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
    
    except Exception as e:
        print(f"Error during GDS to SQLite conversion: {str(e)}")

if __name__ == "__main__":
    INPUT_GDS = "GDS Files/polygon.gds"
    OUTPUT_DB = "DB Files/polygon.db"
    LAYER_NUMBER = 1
    
    gds_to_sqlite(INPUT_GDS, OUTPUT_DB, LAYER_NUMBER)