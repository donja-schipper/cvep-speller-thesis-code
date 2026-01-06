"""
Adapted from https://github.com/thijor/dp-cvep-speller (Thielen et al., 2021).
Modified to include different contrast levels.
"""
import re
from PIL import Image, ImageDraw
from pathlib import Path

WIDTH = 150
HEIGHT = 150

TEXT_COLOR = (128, 128, 128)
FONT_SIZE = 30

CONTRAST_LEVELS = [1.00, 0.60, 0.40, 0.30, 0.20, 0.10]
L0 = 127
AMPLITUDE = 1.0

KEYS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z", ".", ",", "?", "!", "space", "backspace"]

GREEN = (0,128,0)
BLUE = (0,0,255)

# Windows does not allow / , : * ? " < > | ~ in file names
KEY_MAPPING = {
    "comma":",",
    "question":"?",
    "backspace": "<-",
    "exclamation": "!",
    "dot": "."
}

IMAGES_BASE = (Path(__file__).resolve().parent.parent / "images").resolve()
IMAGES_BASE.mkdir(parents=True, exist_ok=True)

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

def save_classic_img(key: str, contrast:int, color:str, img):
    out_dir = IMAGES_BASE / "classic" / f"contrast-{contrast}"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = sanitize(key)
    out_path = out_dir / f"{key}_{color}.png"
    img.save(out_path)
    print("Saved (classic):", out_path)

def save_cue_to_all_classic_contrasts(key:str, img_green, img_blue):
    for c in CONTRAST_LEVELS:
        c_int = int(round(c*100))
        save_classic_img(key, contrast=c_int, color="green", img=img_green)
        save_classic_img(key, contrast=c_int, color="blue", img=img_blue)

for key in KEYS:
    if key in KEY_MAPPING:
         symbol = KEY_MAPPING[key]
    else:
        symbol = key
    for contrast in CONTRAST_LEVELS:
        alpha = contrast * AMPLITUDE
        L_on = max(0, min(255, int(round(L0 * (1 + alpha)))))
        L_off = max(0, min(255, int(round(L0 * (1 - alpha)))))
        ON_BG = (L_on, L_on, L_on)
        OFF_BG = (L_off, L_off, L_off)

        if key == "space":
            img_on = Image.new("RGB", (WIDTH, HEIGHT), ON_BG)
            save_classic_img(key, contrast= int(round(contrast*100)), color="white", img=img_on)
            img_off = Image.new("RGB", (WIDTH, HEIGHT), OFF_BG) 
            save_classic_img(key, contrast= int(round(contrast*100)), color="black", img=img_off)
        else:
            # uppercase on white
            img = Image.new("RGB", (WIDTH, HEIGHT), ON_BG)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol, font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text((x_pos, y_pos), symbol, font_size=FONT_SIZE, fill=TEXT_COLOR)
            save_classic_img(key, contrast= int(round(contrast*100)), color="white", img=img)

            #uppercase on black
            img = Image.new("RGB", (WIDTH, HEIGHT), OFF_BG)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol, font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text((x_pos, y_pos), symbol, font_size=FONT_SIZE, fill=TEXT_COLOR)
            save_classic_img(key, contrast= int(round(contrast*100)), color="black", img=img)

        if key == "space":
            img_green = Image.new("RGB", (WIDTH, HEIGHT), GREEN)
            img_blue = Image.new("RGB", (WIDTH, HEIGHT), BLUE)
            save_classic_img(key, contrast= int(round(contrast*100)), color="green", img=img_green)
            save_classic_img(key, contrast= int(round(contrast*100)), color="blue", img=img_blue)
        else:
            #uppercase on green
            img = Image.new("RGB", (WIDTH, HEIGHT), GREEN)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol, font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text((x_pos, y_pos), symbol, font_size=FONT_SIZE, fill=TEXT_COLOR)
            save_classic_img(key, contrast= int(round(contrast*100)), color="green", img=img)
            
            #uppercase on blue
            img = Image.new("RGB", (WIDTH, HEIGHT), BLUE)
            img_draw = ImageDraw.Draw(img)
            _, _, text_width, text_height = img_draw.textbbox(
                xy=(0, 0), text=symbol, font_size=FONT_SIZE
            )
            x_pos = (WIDTH - text_width) / 2
            y_pos = (HEIGHT - text_height) / 2
            img_draw.text((x_pos, y_pos), symbol, font_size=FONT_SIZE, fill=TEXT_COLOR)
            save_classic_img(key, contrast= int(round(contrast*100)), color="blue", img=img)