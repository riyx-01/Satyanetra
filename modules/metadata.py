from PIL import Image
from PIL.ExifTags import TAGS
import os

def make_serializable(value):
    if hasattr(value, 'numerator') and hasattr(value, 'denominator'):
        return float(value)
    elif isinstance(value, bytes):
        return value.decode(errors='ignore')
    elif isinstance(value, tuple):
        return [make_serializable(v) for v in value]
    elif isinstance(value, dict):
        return {make_serializable(k): make_serializable(v) for k, v in value.items()}
    else:
        try:
            str(value)
            return value
        except Exception:
            return repr(value)

def extract_metadata(image_path):
    metadata = {}
    try:
        image = Image.open(image_path)
        info = image._getexif()

        if info:
            for tag, value in info.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name != "MakerNote":  # ❌ Remove MakerNote
                    metadata[tag_name] = make_serializable(value)

        # Add fallback data
        metadata["Format"] = image.format
        metadata["Mode"] = image.mode
        metadata["Size"] = image.size
        metadata["FileSize (KB)"] = round(os.path.getsize(image_path) / 1024, 2)

    except Exception as e:
        metadata["Error"] = str(e)

    return metadata
