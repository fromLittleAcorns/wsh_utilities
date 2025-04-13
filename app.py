import gradio as gr
import pandas as pd
import numpy as np
import os
from wsh.visulizer import create_dataframe_loader_app

# Create a simple demo with default settings (no initial data)
demo = create_dataframe_loader_app()

# Launch the app
if __name__ == "__main__":
    demo.launch()