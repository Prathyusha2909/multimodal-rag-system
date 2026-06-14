from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "screenshots"
OUTPUT = ROOT / "demo" / "demo.mp4"
SLIDES = [
    ("home-page.png", "Upload PDFs, scans, charts, tables, and images"),
    ("query-example.png", "Ask one question across every modality"),
    ("result-example.png", "Inspect grounded answers and page-level citations"),
]


def title_font(size=38):
    return ImageFont.truetype("C:/Windows/Fonts/segoeuib.ttf", size)


def frame_for(image: Image.Image, caption: str, progress: float) -> np.ndarray:
    width, height = 1280, 720
    scale = 1.0 + progress * 0.025
    target_width = int(width * scale)
    target_height = int(height * scale)
    shot = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    left = (target_width - width) // 2
    top = (target_height - height) // 2
    shot = shot.crop((left, top, left + width, top + height))
    overlay = Image.new("RGBA", shot.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rounded_rectangle((60, 605, 1220, 684), radius=16, fill=(5, 12, 20, 226), outline=(184, 238, 69, 130), width=2)
    draw.text((640, 645), caption, font=title_font(27), fill="#eef4fb", anchor="mm")
    return np.asarray(Image.alpha_composite(shot.convert("RGBA"), overlay).convert("RGB"))


def main():
    OUTPUT.parent.mkdir(exist_ok=True)
    fps = 24
    with imageio.get_writer(OUTPUT, fps=fps, codec="libx264", quality=8, macro_block_size=None) as writer:
        for filename, caption in SLIDES:
            source = Image.open(SHOTS / filename).convert("RGB")
            for index in range(fps * 3):
                writer.append_data(frame_for(source, caption, index / (fps * 3 - 1)))
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()

