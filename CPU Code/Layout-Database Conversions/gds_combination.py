import gdspy
import glob

def combine_gds_files(input_pattern, output_gds, merge_overlaps=True):
    # Create a new library for the combined layout
    combined_lib = gdspy.GdsLibrary()
    combined_cell = gdspy.Cell("COMBINED")
    
    # Find all input GDS files
    input_files = glob.glob(input_pattern)
    
    # Dictionary to store polygons by layer/datatype
    polygons_by_layer = {}
    
    for file in input_files:
        print(f"Processing file: {file}")
        # Load the input GDS file
        gdsii = gdspy.GdsLibrary()
        gdsii.read_gds(file)
        
        # Get the top cell of the input file
        input_cell = gdsii.top_level()[0]
        
        # Get polygons by specification
        polygons_by_spec = input_cell.get_polygons(by_spec=True)
        
        # Store polygons by layer for later processing
        for (layer, datatype), polys in polygons_by_spec.items():
            key = (layer, datatype)
            if key not in polygons_by_layer:
                polygons_by_layer[key] = []
            
            for poly in polys:
                polygons_by_layer[key].append(poly)
    
    # Process polygons by layer
    for (layer, datatype), polys in polygons_by_layer.items():
        if merge_overlaps and len(polys) > 1:
            # Create polygon objects
            poly_objs = [gdspy.Polygon(points, layer=layer, datatype=datatype) for points in polys]
            
            # Merge overlapping polygons using boolean OR operation
            merged_poly = poly_objs[0]
            for poly in poly_objs[1:]:
                merged_poly = gdspy.boolean(merged_poly, poly, 'or')
            
            # Add merged result to the combined cell
            combined_cell.add(merged_poly)
        else:
            # Add individual polygons without merging
            for poly in polys:
                combined_cell.add(gdspy.Polygon(poly, layer=layer, datatype=datatype))
    
    # Add the combined cell to the library
    combined_lib.add(combined_cell)
    
    # Save the combined layout as a new GDS file
    combined_lib.write_gds(output_gds)
    
    print(f"Combined {len(input_files)} GDS files into {output_gds}")

# Usage example
input_pattern = "GDS Files/Divided/small_layout_block_*.gds"
output_gds = "GDS Files/combined_layout.gds"

combine_gds_files(input_pattern, output_gds)