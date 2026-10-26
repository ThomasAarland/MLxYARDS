"""Encode the captured browser walkthrough; requires Pillow and ffmpeg on PATH."""
from pathlib import Path
import shutil
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAPTIONS = [
    "ML x YARDS | NFL expected yards | DAT158 | Thomas Aarland",
    "1. Set a situation. The field shows the ball, line to gain and expected gain.",
    "2. Select PASS. Expected yards changes from 4.82 to 7.29 for this situation.",
    "3. Move to the opponent's 1. Distance becomes goal-to-go and predictions update.",
    "4. Change down, score and clock. Third-and-goal, trailing by 1, Q4 at 2:00.",
    "5. Inspect real results: select on 2024; evaluate the frozen winner on 2025.",
    "6. Compare predictions with outcomes. Individual plays vary much more than means.",
    "7. Inspect residuals and errors by down. Pass plays have larger errors.",
    "8. Inspect global feature importance. Field position and play choice matter most.",
    "9. Read the method and sources. Historical comparisons are not causal advice.",
]


def main():
    if not shutil.which("ffmpeg"):
        raise SystemExit("Install ffmpeg and make it available on PATH to encode this optional walkthrough.")
    # DejaVu is common on Linux; Arial is common on Windows. Fall back portably.
    font = None
    for candidate in ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if Path(candidate).exists():
            font = ImageFont.truetype(candidate, 22)
            break
    font = font or ImageFont.load_default(size=22)
    with tempfile.TemporaryDirectory(dir=ROOT / ".tmp") as directory:
        work = Path(directory)
        for i, caption in enumerate(CAPTIONS, 1):
            screenshot = Image.open(ROOT / "docs/demo_frames" / f"{i:02}.png").convert("RGB")
            frame = Image.new("RGB", (screenshot.width, screenshot.height + 80), "#0b1115")
            frame.paste(screenshot)
            draw = ImageDraw.Draw(frame)
            draw.line((0, screenshot.height, screenshot.width, screenshot.height), fill="#c9ff68", width=2)
            draw.text((28, screenshot.height + 27), caption, font=font, fill="#eef3ed")
            frame.save(work / f"{i:02}.png")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1/6",
            "-start_number", "1", "-i", str(work / "%02d.png"), "-c:v", "libx264", "-r", "24",
            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(ROOT / "docs/demo.mp4")
        ], check=True)
    print("Created docs/demo.mp4 (60-second captioned walkthrough).")


if __name__ == "__main__":
    (ROOT / ".tmp").mkdir(exist_ok=True)
    main()
