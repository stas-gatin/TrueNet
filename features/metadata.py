from PIL import Image, ExifTags

def get_image_metadata(image_path: str) -> dict:
    """
    Extracts EXIF metadata out of an identified image file location path.
    
    Args:
        image_path (str): The provided string URL or os path to image.
        
    Returns:
        dict: Values and items extracted from EXIF tags corresponding properties key value pairs.
    """
    metadata = {}
    try:
        with Image.open(image_path) as img:
            print(f"Format: {img.format}, Mode: {img.mode}")  # Validate format output schema
            exif_data = img.getexif()  # better approach rather than _getexif() legacy functionality implementations
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    metadata[tag] = value
            else:
                print("No EXIF data found")
    except Exception as e:
        print(f"Error: {e}")
    return metadata

if __name__ == "__main__":
    meta = get_image_metadata("/Users/stanislavgatin/Downloads/unnamed-2.png")
    print(meta)
