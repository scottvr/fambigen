import os
import re
import html

# --- CONFIGURATION ---
# The directory where your final PNG ambigrams are stored.
# This should match the strategy you want to display.
IMAGE_DIR = "." 
OUTPUT_HTML_FILE = "gallery.html"
# ---------------------


def parse_filename(filename):
    """
    Parses the complex filename using a regular expression to robustly
    extract the words and the full font name.
    Example: "deadbeef-deadbeef_Colbert_.ttf_ambigram.png"
    """
    pattern = re.compile(r"^([^-]+?)-([^_]+?)_(.*?)\.ttf.*\.png$", re.I)
    
    match = pattern.search(filename)
    
    if not match:
        return None

    word1_part = match.group(1)
    word2_part = match.group(2)
    font_name_part = match.group(3)
    
    # Format for display
    display_words = f"{word1_part}{'/' + word2_part if word1_part not in word2_part else ''}"
    display_font = font_name_part.replace('_', ' ').capitalize()
    
    return {
        'font_name': display_font,
        'words': display_words,
        'full_path': os.path.join(IMAGE_DIR, filename)
    }

def create_html_gallery():
    """
    Scans the image directory and generates a self-contained HTML gallery file.
    """
    print(f"Scanning for images in '{IMAGE_DIR}'...")
    
    # Find all PNG files in the specified directory
    try:
        filenames = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')]
    except FileNotFoundError:
        print(f"ERROR: Directory not found: '{IMAGE_DIR}'. Please ensure this is the correct path.")
        return

    if not filenames:
        print(f"No PNG images found in '{IMAGE_DIR}'.")
        return

    image_data = [parse_filename(f) for f in filenames]
    # Filter out any filenames that couldn't be parsed
    image_data = sorted([d for d in image_data if d], key=lambda x: x['font_name'])

    print(f"Found {len(image_data)} valid images. Generating '{OUTPUT_HTML_FILE}'...")

    # --- HTML and CSS Structure ---
    # We embed the CSS inside the HTML file for simplicity.
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ambigram Typeface Gallery</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 840px;
            margin: 0 auto;
        }}
        header {{
            text-align: center;
            margin-bottom: 3rem;
            border-bottom: 1px solid #333;
            padding-bottom: 1rem;
        }}
        h1 {{
            font-weight: 300;
            font-size: 2.5rem;
            margin: 0;
        }}
        h1 span {{
            font-weight: 600;
        }}
        .gallery-item {{
            margin-bottom: 4rem;
            text-align: center;
        }}
        .gallery-item img {{
            max-width: 100%;
            width: 800px;
            height: auto;
            background-color: #fff; /* White background for the PNGs */
            border-radius: 4px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.5);
            image-rendering: -webkit-optimize-contrast; /* Helps keep thin lines sharp */
            image-rendering: crisp-edges;
        }}
        .caption {{
            margin-top: 1rem;
        }}
        .font-name {{
            font-size: 1.5rem;
            font-weight: 500;
            color: #fff;
        }}
        .words {{
            font-size: 1rem;
            color: #888;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Ambigram Typeface <span>Gallery</span></h1>
        </header>
        <main>
            {image_blocks}
        </main>
    </div>
</body>
</html>
    """

    image_blocks = []
    for data in image_data:
        # Use html.escape to prevent issues with special characters in filenames
        safe_path = html.escape(data['full_path'])
        safe_font_name = html.escape(data['font_name'])
        safe_words = html.escape(data['words'])

        block = f"""
            <div class="gallery-item">
                <img src="{safe_path}" alt="Ambigram for {safe_words} in {safe_font_name}">
                <div class="caption">
                    <div class="font-name">{safe_font_name}</div>
                    <div class="words">{safe_words}</div>
                </div>
            </div>
        """
        image_blocks.append(block)

    # Combine everything and write the final HTML file
    final_html = html_template.format(image_blocks="".join(image_blocks))
    with open(OUTPUT_HTML_FILE, 'w', encoding='utf-8') as f:
        f.write(final_html)
        
    print(f"Gallery saved successfully to '{OUTPUT_HTML_FILE}'.")


if __name__ == '__main__':
    create_html_gallery()