import gradio as gr
import pandas as pd
import numpy as np
import os
from wsh.visulizer import create_dataframe_loader_app

# Define wsh column names
column_mapping = {
    "Name": "Name",
    "Brands": "Brand",
    "Categories": "Category",
    "Short description": "Description",
    "Material_new": "Material",
    "Colour_new": "Color",
    "Finish_new": "Finish",
    "Weight new": "Weight",
    "Dimension_new": "Dimensions",
    "Pack quantity": "Pack quantity",
    "Parts": "Number of parts",
    "Fire rated?": "Fire Rated?",
    "Certified": "Certification",
    "summary": "LLM Input",
}

# Define layouts
left_fields = ["Fire rated?", "Certified", "Guarantee", "Number of parts"]
right_fields = ["Name", "Brands", "Short description", "Categories", "Colour_new", 
                "Finish_new", "Material_new", "Dimensions_new", "Weight new", "Pack quantity"]

# Define editable fields
editable = ["Brands", "Short description", "Colour_new", "Finish_new", "Material_new", "Dimensions_new", "Pack quantity", 
           "Parts", "Fire rated?", "Certified", "Guarantee", "Name", "Weight new", "Categories"]

# Create a visuliser
vis = create_dataframe_loader_app(
    initial_image_dir=None,
    column_display_names=column_mapping,
    left_column_fields=left_fields,
    right_column_fields=right_fields,
    main_field="summary",  # The LLM input field
    editable_fields=editable,
    max_image_height=350,  # Restrict image height
    compact_mode=True,      # Enable compact layout
    cs_out_visible=True
)

# Launch the app
if __name__ == "__main__":
    vis.launch(server_name="0.0.0.0", root_path="/vis", share=False)
