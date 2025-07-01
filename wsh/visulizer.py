import gradio as gr
import pandas as pd
import numpy as np
import os
import tempfile
from pathlib import Path
import shutil
import traceback
from typing import Dict, List, Tuple, Union, Optional, Any

from wsh.vis_utils import resize_image_with_max_side_length, get_image_from_url, local_url_to_image, resize_image_with_max_height

def create_dataframe_loader_app(
    initial_df=None,
    initial_image_dir=None,
    column_display_names=None,
    left_column_fields=None,
    right_column_fields=None,
    main_field=None,
    editable_fields=None,
    image_column='Images',
    max_image_height=400,
    compact_mode=True,
    cs_out_visible=True  # New parameter to control visibility of cs_out field
):
    """
    Create a Gradio app with dataframe loading, editing, and saving capabilities.
    
    Args:
        initial_df: Initial dataframe to display (optional)
        initial_image_dir: Directory with images (optional)
        column_display_names: Dictionary mapping column names to display labels
        left_column_fields: List of columns to show in left column (below image)
        right_column_fields: List of columns to show in right column
        main_field: Field to display as main content (typically LLM input/summary)
        editable_fields: List of fields that should be editable
        image_column: Column name containing image URLs if using URL images
        max_image_height: Maximum height for image display
        compact_mode: Enable compact layout with reduced spacing
        cs_out_visible: Whether to display the cs_out field (defaults to True)
    
    Returns:
        Gradio Blocks interface
    """
    print("\n==================================")
    print("create_dataframe_loader_app called")
    print(f"initial_df = {type(initial_df)}")
    print(f"left_column_fields = {left_column_fields}")
    print(f"right_column_fields = {right_column_fields}")
    print(f"main_field = {main_field}")
    print(f"editable_fields = {editable_fields}")
    print("==================================\n")
    
    # Create a temporary directory for file uploads and downloads
    temp_dir = tempfile.mkdtemp()
    
    # Global state that will be accessed within functions
    class AppState:
        df = initial_df
        image_dir = initial_image_dir
        
    state = AppState()
    
    # If no column mappings are provided, use column names directly
    if column_display_names is None:
        column_display_names = {}
    
    # Ensure editable_fields is a list, not None
    if editable_fields is None:
        editable_fields = []
    
    # Ensure left_column_fields is a list, not None
    if left_column_fields is None:
        left_column_fields = []
    
    # Ensure right_column_fields is a list, not None
    if right_column_fields is None:
        right_column_fields = []
    
    # Determine fields to display initially
    all_fields = []
    
    # Build the list of all fields in order, even if we don't have initial_df
    if main_field:
        all_fields.append(main_field)
    
    all_fields.extend([f for f in left_column_fields if f != main_field])
    all_fields.extend([f for f in right_column_fields if f != main_field])
    
    # Make sure cs_out is not in all_fields (it will be handled separately)
    all_fields = [f for f in all_fields if f != "cs_out"]
    
    print(f"All fields: {all_fields}")
    
    # Function to process an uploaded dataframe file
    def process_dataframe(file):
        if file is None:
            return "No file uploaded", gr.update(choices=[], value=None), gr.update(visible=True), gr.update(visible=False)
        
        try:
            print(f"Processing file: {file.name}")
            file_ext = os.path.splitext(file.name)[1].lower()
            if file_ext == '.csv':
                df = pd.read_csv(file.name)
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file.name)
            elif file_ext == '.pkl':
                df = pd.read_pickle(file.name)
            else:
                return f"Unsupported file format: {file_ext}", gr.update(choices=[], value=None), gr.update(visible=True), gr.update(visible=False)
            
            print(f"DataFrame loaded with columns: {df.columns.tolist()}")
            print(f"DataFrame index: {df.index.name}")
            
            # Handle SKU column
            if 'SKU' in df.columns:
                print("SKU column found, converting to string and setting as index")
                # Convert SKU column to string type first
                df['SKU'] = df['SKU'].astype(str)
                
                # Then set as index if not already
                if df.index.name != 'SKU':
                    df = df.set_index('SKU')
            elif df.index.name == 'SKU':
                print("SKU is already the index, ensuring it's string type")
                # If SKU is already the index, ensure it's string type
                df.index = df.index.astype(str)
            else:
                print("No SKU column found, using existing index as string")
                # If no SKU column or index, use the existing index but convert to string
                df.index = df.index.astype(str)
            
            # Add any missing columns from our field definitions
            for field in all_fields:
                if field not in df.columns:
                    print(f"Adding missing column: {field}")
                    df[field] = None
            
            # Add cs_out column if it doesn't exist and we're showing it
            if cs_out_visible and "cs_out" not in df.columns:
                print("Adding missing cs_out column")
                df["cs_out"] = None
                
            state.df = df
            
            # Get choices as strings
            choices = df.index.tolist()
            if choices:
                print(f"First SKU in choices: {choices[0]}")
                print(f"Number of SKUs: {len(choices)}")
            else:
                print("No SKUs found in dataframe")
                
            # Update the SKU dropdown and switch tabs
            first_sku = choices[0] if choices else None
            return (
                f"Successfully loaded dataframe with {len(df)} rows and {len(df.columns)} columns", 
                gr.update(choices=choices, value=first_sku),
                gr.update(visible=False),  # Hide no-data message
                gr.update(visible=True),   # Show visualization container
                first_sku                  # Return the first SKU as an additional value
            )
        except Exception as e:
            print(f"Error loading file: {str(e)}")
            traceback.print_exc()
            return f"Error loading file: {str(e)}", gr.update(choices=[], value=None), gr.update(visible=True), gr.update(visible=False)    
    
    # Function to process uploaded images
    def process_images(files):
        if not files:
            return "No images uploaded"
        
        try:
            # Create a directory for the uploaded images
            image_dir = os.path.join(temp_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            
            count = 0
            for file in files:
                try:
                    # Get the filename (should be SKU.ext)
                    filename = os.path.basename(file.name)
                    sku = os.path.splitext(filename)[0]
                    
                    # Process and save the image
                    img = local_url_to_image(file.name)
                    img.save(os.path.join(image_dir, f"{sku}.png"))
                    count += 1
                except Exception as e:
                    print(f"Error processing image {file.name}: {e}")
            
            state.image_dir = image_dir
            return f"Successfully processed {count} images"
        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return f"Error processing images: {str(e)}"
    
    # Function to save the dataframe
    def save_dataframe(format_choice, custom_filename=None):
        if state.df is None:
            return "No dataframe to save", None
        
        try:
            # Determine the correct file extension based on format_choice
            if format_choice == 'excel':
                file_extension = 'xlsx'
            else:
                file_extension = format_choice
            
            # Use custom filename if provided, otherwise use "dataframe"
            base_filename = custom_filename if custom_filename and custom_filename.strip() else "dataframe"
            
            # Make sure the filename is safe for filesystem use
            base_filename = "".join(c for c in base_filename if c.isalnum() or c in "._- ")
            
            # Create the output path
            output_path = os.path.join(temp_dir, f"{base_filename}.{file_extension}")
            
            # Save the dataframe in the selected format
            if format_choice == 'csv':
                state.df.to_csv(output_path)
            elif format_choice == 'excel':
                state.df.to_excel(output_path)
            elif format_choice == 'pkl':
                state.df.to_pickle(output_path)
            else:
                return f"Unsupported format: {format_choice}", None
            
            return f"Dataframe saved successfully as {os.path.basename(output_path)}", output_path
        except Exception as e:
            print(f"Error saving dataframe: {str(e)}")
            return f"Error saving dataframe: {str(e)}", None
        
    # Function to update any field in the dataframe
    def update_field(sku, field, value):
        if state.df is None or sku not in state.df.index:
            return f"Error: Cannot update {field} for SKU {sku}"
        
        try:
            state.df.loc[sku, field] = value
            return f"Updated {field} for SKU {sku}"
        except Exception as e:
            print(f"Error updating field: {str(e)}")
            return f"Error updating field: {str(e)}"
    
    # Get details and image for a specific SKU
    def get_part_details(sku):
        print(f"\nget_part_details called for SKU: {sku}")
        
        if state.df is None:
            error_msg = "No dataframe loaded"
            print(error_msg)
            # If no dataframe is loaded, return None for all fields
            result = [None] * (len(all_field_boxes) + 3)  # +3 for image, cs_out, and error box
            result[-1] = error_msg  # Set error message in the last position
            return result
            
        if not sku:
            error_msg = "No SKU selected"
            print(error_msg)
            # If no SKU is selected, return None for all fields
            result = [None] * (len(all_field_boxes) + 3)
            result[-1] = error_msg
            return result
            
        if sku not in state.df.index:
            error_msg = f"SKU {sku} not found in dataframe"
            print(error_msg)
            # If SKU is not in dataframe, return None for all fields
            result = [None] * (len(all_field_boxes) + 3)
            result[-1] = error_msg
            return result
        
        try:
            print(f"Getting details for SKU: {sku}")
            # Handle image loading
            img = None
            if state.image_dir is not None:
                # Try to get image from local directory
                local_image_url = os.path.join(state.image_dir, f"{sku}.png")
                if os.path.exists(local_image_url):
                    print(f"Loading image from: {local_image_url}")
                    img = local_url_to_image(local_image_url)
            elif image_column in state.df.columns:
                # Try to get image from URL in the dataframe
                image_url = state.df.loc[sku, image_column]
                if isinstance(image_url, str) and image_url.strip():
                    print(f"Loading image from URL: {image_url}")
                    img = get_image_from_url(image_url)
            
            if img:
                img = resize_image_with_max_side_length(img, max_image_height)
            
            # Build results for fields in the order they were created
            results = [img]  # Start with image
            
            for field in all_fields:
                if field in state.df.columns:
                    value = state.df.loc[sku, field]
                    print(f"Field {field}: {value}")
                    results.append(value)
                else:
                    print(f"Field {field} not found in dataframe")
                    results.append(None)
            
            # Add the cs_out field if we're showing it
            if cs_out_visible:
                if "cs_out" in state.df.columns:
                    cs_out_value = state.df.loc[sku, "cs_out"]
                    print(f"cs_out: {cs_out_value}")
                    results.append(cs_out_value)
                else:
                    print("cs_out field not found in dataframe")
                    results.append(None)
            
            # Add None for error box (no error)
            results.append(None)
            
            print(f"get_part_details returning {len(results)} results")
            return results
        except Exception as e:
            # Error handling - return None for all fields except error
            error_msg = f"Error loading details for SKU {sku}: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            results = [None] * (len(all_field_boxes) + 3)
            results[-1] = error_msg
            return results
    
    # Navigation functions
    def prev_sku(current_sku):
        print(f"prev_sku called for {current_sku}")
        if state.df is None or not current_sku:
            return current_sku, *([None] * (len(all_field_boxes) + 2))
        
        sku_choices = state.df.index.tolist()
        try:
            current_index = sku_choices.index(current_sku)
            
            if current_index > 0:
                new_sku = sku_choices[current_index - 1]
                print(f"Moving to previous SKU: {new_sku}")
            else:
                new_sku = current_sku
                print(f"Already at first SKU: {new_sku}")
            
            results = get_part_details(new_sku)
            return new_sku, *results
        except (ValueError, IndexError) as e:
            # Handle case where current_sku is not in choices
            error_msg = f"Error navigating: {str(e)}"
            print(error_msg)
            results = [None] * (len(all_field_boxes) + 3)
            results[-1] = error_msg
            return current_sku, *results
    
    def next_sku(current_sku):
        print(f"next_sku called for {current_sku}")
        if state.df is None or not current_sku:
            return current_sku, *([None] * (len(all_field_boxes) + 2))
        
        sku_choices = state.df.index.tolist()
        try:
            current_index = sku_choices.index(current_sku)
            
            if current_index < len(sku_choices) - 1:
                new_sku = sku_choices[current_index + 1]
                print(f"Moving to next SKU: {new_sku}")
            else:
                new_sku = current_sku
                print(f"Already at last SKU: {new_sku}")
            
            results = get_part_details(new_sku)
            return new_sku, *results
        except (ValueError, IndexError) as e:
            # Handle case where current_sku is not in choices
            error_msg = f"Error navigating: {str(e)}"
            print(error_msg)
            results = [None] * (len(all_field_boxes) + 3)
            results[-1] = error_msg
            return current_sku, *results
    
    def get_first_sku_details():
        if state.df is None or len(state.df.index) == 0:
            return [None] * len(output_components)
        
        first_sku = state.df.index[0]
        print(f"Loading first SKU details for: {first_sku}")
        return get_part_details(first_sku)

    # Create the Gradio interface
    # Set theme based on compact mode
    if compact_mode:
        theme = gr.themes.Default(
            spacing_size=gr.themes.sizes.spacing_sm,
            radius_size=gr.themes.sizes.radius_none,
            text_size=gr.themes.sizes.text_sm
        )
    else:
        theme = gr.themes.Default()

    # Add custom CSS for compact mode
    custom_css = """
    .container {
        margin-bottom: 6px !important;
        padding: 2px !important;
    }
    label {
        margin-bottom: 2px !important;
    }
    textarea {
        min-height: 20px !important;
        padding: 4px !important;
    }
    .gradio-row {
        margin-bottom: 6px !important;
    }
    """ if compact_mode else ""

    with gr.Blocks(theme=theme, css=custom_css) as demo:
        # Tab structure
        with gr.Tabs() as tabs:
            # Tab 1: Load and Save Data
            with gr.TabItem("Load & Save Data") as load_tab:
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Upload Data")
                        upload_file = gr.File(label="Upload DataFrame (CSV, Excel, Pickle)")
                        upload_button = gr.Button("Load DataFrame", variant="primary")
                        upload_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Upload Images (Optional)")
                        gr.Markdown("*Image filenames should match SKUs (e.g., SKU123.png)*")
                        image_files = gr.File(label="Upload Images", file_count="multiple")
                        upload_images_button = gr.Button("Process Images")
                        image_status = gr.Textbox(label="Image Status", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Save Data")
                        filename_input = gr.Textbox(
                            label="Filename (without extension)",
                            placeholder="dataframe",
                            value="dataframe",
                            interactive=True
)
                        save_format = gr.Radio(
                            choices=["csv", "excel", "pkl"],
                            value="csv",
                            label="Save Format"
                        )
                        save_button = gr.Button("Save DataFrame")
                        save_status = gr.Textbox(label="Save Status", interactive=False)
                        download_file = gr.File(label="Download DataFrame", interactive=False)
            
            # Tab 2: Visualize and Edit
            with gr.TabItem("Visualize & Edit") as vis_tab:
                # Message shown when no data is loaded
                no_data_message = gr.Markdown(
                    "Please load a dataframe in the 'Load & Save Data' tab first.",
                    visible=True if initial_df is None else False
                )
                
                # Create visualization interface - initially hidden if no data
                with gr.Column(visible=False if initial_df is None else True) as vis_container:
                    # Top row with SKU selector and navigation buttons
                    with gr.Row():
                        with gr.Column(scale=2, min_width=200):
                            sku_choices = [] if initial_df is None else initial_df.index.tolist()
                            def_sku = sku_choices[0] if sku_choices else None
                            sku_dd = gr.Dropdown(
                                choices=sku_choices, 
                                value=def_sku,  # Use def_sku here
                                interactive=True, 
                                label="SKU",
                                allow_custom_value=False
                            )
                            print(f"SKU dropdown created with choices: {sku_choices}")
                            print(f"Default SKU value: {def_sku}")
                        
                        with gr.Column(scale=1, min_width=50):
                            prev_btn = gr.Button("Previous")
                            next_btn = gr.Button("Next")
                    
                    # Main content area
                    with gr.Row():
                        # Left column with image and fields below it
                        with gr.Column(scale=1, min_width=200):
                            # Image display
                            image_box = gr.Image(label="Image", height=max_image_height)
                            
                            # Fields below image
                            left_field_boxes = {}
                            if compact_mode and left_column_fields:
                                # For compact mode, place left fields in a grid
                                left_field_count = len(left_column_fields)
                                grid_cols = min(2, left_field_count)  # Use at most 2 columns
                                
                                for i in range(0, left_field_count, grid_cols):
                                    with gr.Row():
                                        # Take up to grid_cols fields at a time
                                        fields_chunk = left_column_fields[i:i + grid_cols]
                                        for field in fields_chunk:
                                            with gr.Column(scale=1):
                                                display_name = column_display_names.get(field, field)
                                                is_editable = field in editable_fields
                                                
                                                field_box = gr.Textbox(
                                                    label=display_name, 
                                                    interactive=is_editable,
                                                    lines=1,
                                                    scale=1,
                                                    min_width=100
                                                )
                                                left_field_boxes[field] = field_box
                            else:
                                # Standard layout for left fields
                                for field in left_column_fields:
                                    display_name = column_display_names.get(field, field)
                                    is_editable = field in editable_fields
                                    
                                    field_box = gr.Textbox(
                                        label=display_name, 
                                        interactive=is_editable,
                                        lines=1 if compact_mode else 2
                                    )
                                    left_field_boxes[field] = field_box
                        
                        # Right column with remaining fields
                        with gr.Column(scale=2, min_width=200):
                            # Main field (summary/LLM input) at the top if specified
                            main_box = None
                            if main_field:
                                display_name = column_display_names.get(main_field, main_field)
                                is_editable = main_field in editable_fields
                                
                                main_box = gr.Textbox(
                                    label=display_name,
                                    interactive=is_editable,
                                    lines=3 if compact_mode else 5  # Make it taller
                                )
                            
                            # Other fields in the right column
                            right_field_boxes = {}
                            if compact_mode and right_column_fields:
                                # Filter out main field if it exists
                                right_fields = [f for f in right_column_fields if f != main_field]
                                # For compact mode, arrange right fields in a grid
                                right_field_count = len(right_fields)
                                grid_cols = min(2, right_field_count)  # Use at most 2 columns
                                
                                for i in range(0, right_field_count, grid_cols):
                                    with gr.Row():
                                        # Take up to grid_cols fields at a time
                                        fields_chunk = right_fields[i:i + grid_cols]
                                        for field in fields_chunk:
                                            with gr.Column(scale=1):
                                                display_name = column_display_names.get(field, field)
                                                is_editable = field in editable_fields
                                                
                                                field_box = gr.Textbox(
                                                    label=display_name, 
                                                    interactive=is_editable,
                                                    lines=1,
                                                    scale=1,
                                                    min_width=100
                                                )
                                                right_field_boxes[field] = field_box
                            else:
                                # Standard layout for right fields
                                for field in right_column_fields:
                                    # Skip if this was already the main field
                                    if field == main_field:
                                        continue
                                    
                                    display_name = column_display_names.get(field, field)
                                    is_editable = field in editable_fields
                                    
                                    field_box = gr.Textbox(
                                        label=display_name, 
                                        interactive=is_editable,
                                        lines=1 if compact_mode else 2
                                    )
                                    right_field_boxes[field] = field_box
                    
                    # Status/error box at bottom
                    error_box = gr.Textbox(label="Status/Errors")
                    
                    # Add cs_out field below error box, spanning full width
                    cs_out_box = None
                    if cs_out_visible:
                        with gr.Row():
                            with gr.Column(scale=1, min_width=400):
                                cs_out_box = gr.Textbox(
                                    label="LLM Output",
                                    interactive="cs_out" in editable_fields,
                                    lines=3 if compact_mode else 5,  # Make it taller like the main field
                                )
        
        # Combine all field boxes in order
        all_field_boxes = {}
        if main_field and main_box:
            all_field_boxes[main_field] = main_box
        all_field_boxes.update(left_field_boxes)
        all_field_boxes.update(right_field_boxes)
        
        print(f"Created field boxes: {list(all_field_boxes.keys())}")
        
        # Create output lists for event handlers
        output_components = [image_box]
        for field in all_fields:
            if field in all_field_boxes:
                output_components.append(all_field_boxes[field])
                
        # Add cs_out to outputs if it's visible
        if cs_out_visible and cs_out_box:
            output_components.append(cs_out_box)
                
        output_components.append(error_box)
        
        print(f"Output components count: {len(output_components)}")
        
        # Connect file upload events
        upload_button.click(
            fn=process_dataframe,
            inputs=upload_file,
            outputs=[upload_status, sku_dd, no_data_message, vis_container]
        ).then(
            fn=lambda: gr.update(selected="Visualize & Edit"),  # Switch to visualization tab by name
            inputs=None,
            outputs=tabs
        ).then(
            # Add a small delay to ensure the tab is fully visible before updating components
            fn=lambda: None,  # Do nothing, just add a delay
            inputs=None,
            outputs=None,
            js="() => new Promise(resolve => setTimeout(() => resolve(), 300))"  # 300ms delay
        ).then(
            # Now trigger the details load with the current SKU value
            fn=lambda: get_part_details(state.df.index[0]) if state.df is not None and len(state.df.index) > 0 else [None] * len(output_components),
            inputs=None,
            outputs=output_components
)
        
        upload_images_button.click(
            fn=process_images,
            inputs=image_files,
            outputs=image_status
        )
        
        save_button.click(
            fn=save_dataframe,
            inputs=[save_format, filename_input],
            outputs=[save_status, download_file]
)
        
        # Connect visualization events
        # Always connect the SKU dropdown change event
        sku_dd.change(
            fn=get_part_details,
            inputs=sku_dd,
            outputs=output_components
        )
        
        # Always connect navigation buttons
        prev_btn.click(
            fn=prev_sku,
            inputs=sku_dd,
            outputs=[sku_dd] + output_components
        )
        
        next_btn.click(
            fn=next_sku,
            inputs=sku_dd,
            outputs=[sku_dd] + output_components
        )

        # tabs.change(
        #     fn=lambda tab_name: gr.update(value=state.df.index[0]) if tab_name == "Visualize & Edit" and state.df is not None and len(state.df.index) > 0 else gr.update(),
        #     inputs=tabs,
        #     outputs=sku_dd
        # )
        
        # Connect update events for editable fields
        for field, field_box in all_field_boxes.items():
            if field in editable_fields:
                field_box.change(
                    fn=lambda sku, field_name, value: update_field(sku, field_name, value),
                    inputs=[sku_dd, gr.State(value=field), field_box],
                    outputs=error_box
                )
        
        # Connect update event for cs_out if it's visible and editable
        if cs_out_visible and cs_out_box and "cs_out" in editable_fields:
            cs_out_box.change(
                fn=update_field,
                inputs=[sku_dd, gr.State("cs_out"), cs_out_box],
                outputs=error_box
            )
        
        # If there's an initial dataframe, load the first SKU data on startup
        if initial_df is not None and def_sku is not None:
            demo.load(
                fn=lambda: get_part_details(def_sku),
                inputs=None,
                outputs=output_components
            )

        tabs
    
    return demo
