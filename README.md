---
title: Data Visualizer and Editor
emoji: 📊
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.50.2
app_file: app.py
pinned: false
license: mit
---

# Data Visualizer and Editor

An interactive tool for visualizing and editing dataframes. Upload your data, visualize it with images, and edit fields directly in the browser.

Note that this app is intended to focus upon parts having unique SKU's, as is common in manufacturing and retail.

## Features

- Upload dataframes in CSV, Excel, or Pickle format
- Upload images to associate with SKUs
- View and edit fields with a user-friendly interface
- Save modified dataframes back to your computer
- Customizable layout with support for different column arrangements

## Usage

1. Upload a dataframe file (CSV, Excel, or Pickle)
2. Optionally upload images (file names should match SKUs)
3. Navigate between SKUs using the dropdown or navigation buttons
4. Edit fields as needed
5. Save your modified dataframe

## Setup Locally

Ideally setup a local venv and activate it.  Then:

```bash
pip install -r requirements.txt
python app.py