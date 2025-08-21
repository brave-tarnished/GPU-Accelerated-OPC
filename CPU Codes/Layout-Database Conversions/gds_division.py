import gdspy
import math
import numpy as np

def divide_gds_file(input_gds, output_prefix, divisor):
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
    
    # Divide the layout into blocks
    blocks_created = 0
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
                output_file = f"GDS Files/Divided/{output_prefix}_block_{i}_{j}.gds"
                block_lib.write_gds(output_file)
                print(f"Created block file: {output_file}")
                blocks_created += 1
            else:
                print(f"Block ({i},{j}) is empty, skipping.")
    
    print(f"Divided the layout into {blocks_created} non-empty blocks out of {divisor} possible blocks.")

# Usage example
input_gds = "GDS Files/multiple_polygon.gds"
output_prefix = "small_layout"
divisor = 9

divide_gds_file(input_gds, output_prefix, divisor)