"""
Adapted from https://github.com/thijor/dp-cvep-speller (Thielen et al., 2021).
Modified to include different contrast levels.
"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import re

np.random.seed(2)

WIDTH = 150
HEIGHT = 150

PATCH_HEIGHT = 20
PATCH_WIDTH = 20
N_PATCHES = 75

TEXT_COLOR = (0, 0, 0)
FONT_SIZE = 30
GRAY_COLOR = (127, 127, 127)

#list of the contrast levels
CONTRAST_LEVELS = [1.00, 0.60, 0.40, 0.30, 0.20, 0.10]
AMPLITUDE = 1.0

IMAGES_BASE = (Path(__file__).resolve().parent.parent / "images").resolve()
IMAGES_BASE.mkdir(parents=True, exist_ok=True)

KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z", ".", ",", "?", "!", "space", "backspace"]

# Windows does not allow / , : * ? " < > | ~ in file names
KEY_MAPPING = {
    "comma": ",",
    "question": "?",
    "backspace": "<-",
    "exclamation": "!",
    "dot": "."
}

SPECIALS = {
    "!":"exclam", "@":"at", "#":"hash", "$":"dollar", "%":"percent", "^":"caret",
    "&":"amp", "*":"asterisk", "(":"lparen", ")":"rparen", "_":"underscore",
    "+":"plus", "-":"minus", "=":"equals", "[":"lbrack", "]":"rbrack",
    "{":"lbrace", "}":"rbrace", ":":"colon", ";":"semicolon", "'":"quote",
    '"':"dquote", "\\":"backslash", "/":"slash", "`":"backtick", "~":"tilde",
    ",":"comma", ".":"dot", "<":"smaller", ">":"larger", "?":"question", " ":"space",
    "<-":"backspace"
}

def sanitize(name: str) -> str:
    name = SPECIALS.get(name, name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)

def save_grating_img(key: str, contrast: int, color: str, img):
    out_dir = IMAGES_BASE / "grating" / f"contrast-{contrast}"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = sanitize(key)
    out_path = out_dir / f"{key}_{color}.png"
    img.save(out_path)
    print("Saved:", out_path)

def generate_gabor_patch(
    size=(60, 60), theta=np.pi / 2, gamma=0.6, lamda=5, phi=0.0, sigma=2
):
    x, y = np.meshgrid(
        np.linspace(-size[1] // 2, size[1] // 2, size[1]),
        np.linspace(-size[0] // 2, size[0] // 2, size[0]),
    )

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    gabor = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2)) * np.cos(
        2 * np.pi * x_theta / lamda + phi
    )

    return gabor


grating_image = np.zeros(shape=(WIDTH, HEIGHT), dtype="float32")
for i in range(N_PATCHES):
    patch = generate_gabor_patch(
        size=(PATCH_HEIGHT, PATCH_WIDTH), theta=np.random.rand() * np.pi
    )
    while True:
        x_pos = int(np.random.rand() * (WIDTH - PATCH_WIDTH))
        y_pos = int(np.random.rand() * (HEIGHT - PATCH_HEIGHT))
        if (
            np.sqrt(
                (x_pos + PATCH_WIDTH // 2 - WIDTH // 2) ** 2
                + (y_pos + PATCH_HEIGHT // 2 - HEIGHT // 2) ** 2
            )
            > FONT_SIZE // 2
        ):
            break
    grating_image[y_pos : y_pos + PATCH_HEIGHT, x_pos : x_pos + PATCH_WIDTH] += patch

max_abs = np.max(np.abs(grating_image)) if np.max(np.abs(grating_image)) > 0 else 1.0
grating_unit = grating_image/max_abs

#gray/blue/green comparison image
gray_base = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=GRAY_COLOR)
green_base = Image.new(mode="RGB", size=(WIDTH, HEIGHT), color=(0,128,0))
blue_base = Image.new(mode="RGB", size=(WIDTH, HEIGHT),color=(0,0,255))

for CONTRAST in CONTRAST_LEVELS:
    for key in KEYS:
        symbol = KEY_MAPPING.get(key, key)

        if key == "space":
            img = gray_base.copy()
            save_grating_img(key, contrast=int(round(CONTRAST*100)), color="gray", img=img)
            img_green = green_base.copy()
            save_grating_img(key, contrast=int(round(CONTRAST*100)), color="green", img=img_green)
            img_blue = blue_base.copy()
            save_grating_img(key, contrast=int(round(CONTRAST*100)), color="blue", img=img_blue)
            continue
    
        #uppercase letters
        img = gray_base.copy()
        img_draw = ImageDraw.Draw(img)
        _, _, text_width, text_height = img_draw.textbbox(
            xy=(0, 0), text=symbol, font_size=FONT_SIZE
        )
        x_pos = (WIDTH - text_width) / 2
        y_pos = (HEIGHT - text_height) / 2
        img_draw.text(
            xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
        )
        save_grating_img(key, contrast=int(round(CONTRAST*100)), color="gray", img=img)

        img_green = green_base.copy()
        img_draw = ImageDraw.Draw(img_green)
        _, _, text_width, text_height = img_draw.textbbox(
            xy=(0, 0), text=symbol, font_size=FONT_SIZE
        )
        x_pos = (WIDTH - text_width) / 2
        y_pos = (HEIGHT - text_height) / 2
        img_draw.text(
            xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
        )
        save_grating_img(key, contrast=int(round(CONTRAST*100)), color="green", img=img_green)

        img_blue = blue_base.copy()
        img_draw = ImageDraw.Draw(img_blue)
        _, _, text_width, text_height = img_draw.textbbox(
            xy=(0, 0), text=symbol, font_size=FONT_SIZE
        )
        x_pos = (WIDTH - text_width) / 2
        y_pos = (HEIGHT - text_height) / 2
        img_draw.text(
            xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
        )
        save_grating_img(key, contrast=int(round(CONTRAST*100)), color="blue", img=img_blue)

    #grating
    ALPHA = CONTRAST * AMPLITUDE
    grating_8u = (127 + 127 * ALPHA * grating_unit).clip(0, 255).astype("uint8")
    grating_rgb = np.repeat(grating_8u[:, :, np.newaxis], repeats=3, axis=2)

    for key in KEYS:
        if key == "space":
            img = Image.fromarray(grating_rgb)
            img_draw = ImageDraw.Draw(img)
            save_grating_img(key, contrast=int(round(CONTRAST*100)), color="grating", img=img)

        else:
            if key in KEY_MAPPING:
                symbol = KEY_MAPPING[key]
            else:
                symbol = key
            
            img = Image.fromarray(grating_rgb)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol, font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text(
                xy=(x_pos, y_pos), text=symbol, fill=TEXT_COLOR, font_size=FONT_SIZE
            )
            save_grating_img(key, contrast=int(round(CONTRAST*100)), color="grating", img=img)