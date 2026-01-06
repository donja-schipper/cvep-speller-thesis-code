"""
c-VEP speller implementation used in this thesis.

This code is adapted from the open-source c-VEP speller implementation:
https://github.com/thijor/dp-cvep-speller

Original authors:
- Thielen et al. (2021)

The present implementation reuses the general speller structure and
stimulus presentation logic, but introduces some modifications,
including:
- Support for both classic and Gabor stimulus banks
- Contrast-level manipulation
- Balanced trial scheduling
- Adjustments to stimulus timing and trial structure

This code was used for stimulus presentation during EEG data acquisition.
"""
import json
import os
import random
import sys
import time
from pathlib import Path
import numpy as np
import psychopy
import toml
from dareplane_utils.logging.logger import get_logger
from dareplane_utils.stream_watcher.lsl_stream_watcher import StreamWatcher
from fire import Fire
from psychopy import event, misc, monitors, visual
from pylsl import StreamInfo, StreamOutlet
from cvep_speller.utils.logging import logger as speller_logger
# from collections import defaultdict
# from itertools import cycle

# Windows does not allow / , : * ? " < > | ~ in file names (for the images)
KEY_MAPPING = {
    "space": " ",
    "dot": ".",
    "exclam": "!",
    "slash": "/",
    "comma": ",",
    "question": "?",
    "backspace": "<-",
}

def _get_bank(cfg, condition: str) -> dict:
    """
    Get the bank. There are two banks available: grating and classic. 

    Parameters
    ----------
    cfg: dict
        config object containing context and paradigm configuration info as loaded
        from `./configs/speller.toml`.
    condition: str
        The condition that is being tested. Either 'classic' or 'grating'.
        
    Returns
    -------
    bank: dict
        The bank configuration dictionary for the requested condition.
    """
    for b in cfg["speller"]["banks"]:
        if b["condition"] == condition:
            return b
    raise ValueError(f"No bank found for condition='{condition}'")

class Speller(object):
    """
    An object to create a speller with keys and text fields. Keys can alternate their background images according to
    specifically setup stimulation sequences.

    Parameters
    ----------
    screen_resolution: tuple[int, int]
        The screen resolution in pixels, provided as (width, height).
    width_cm: float
        The width of the screen in cm to compute pixels per degree.
    distance_cm: float
        The distance of the screen to the user in cm to compute pixels per degree.
    refresh_rate: int
        The screen refresh rate in Hz.
    cfg: dict
        config object containing context and paradigm configuration info as loaded
        from `./configs/speller.toml`.
    screen_id: int (default: 0)
        The screen number where to present the keyboard when multiple screens are used.
    background_color: tuple[float, float, float] (default: (0., 0., 0.))
        The keyboard's background color specified as list of RGB values.
    marker_stream_name: str (default: "marker-stream")
        The name of the LSL stream to which markers of the keyboard are logged.
    quit_controls: list[str] (default: None)
        A list of keys that can be pressed to initiate quiting of the speller.
    full_screen: bool (default: True)
        Whether to present the speller in full screen mode.

    Attributes
    ----------
    keys: dict
        A dictionary of keys with a mapping of key name to a list of PsychoPy ImageStim.
    text_fields: dict
        A dictionary of text fields with a mapping of text field name to PsychoPy TextBox2.
    """

    keys: dict = dict()
    text_fields: dict = dict()

    def __init__(
        self,
        screen_resolution: tuple[int, int],
        width_cm: float,
        distance_cm: float,
        refresh_rate: int,
        cfg: dict,
        screen_id: int = 0,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        marker_stream_name: str = "marker-stream",
        quit_controls: list[str] = None,
        full_screen: bool = True,
    ) -> None:
        self.cfg = cfg
        self.screen_resolution = screen_resolution
        self.full_screen = full_screen
        self.width_cm = width_cm
        self.distance_cm = distance_cm
        self.refresh_rate = refresh_rate
        self.quit_controls = quit_controls

        # Setup monitor
        self.monitor = monitors.Monitor(
            name="testMonitor", width=width_cm, distance=distance_cm
        )
        self.monitor.setSizePix(screen_resolution)

        # Setup window
        self.window = visual.Window(
            monitor=self.monitor,
            screen=screen_id,
            units="pix",
            size=screen_resolution,
            color=background_color,
            fullscr=full_screen,
            waitBlanking=False,
            allowGUI=False,
        )
        self.window.setMouseVisible(False)

        # Setup LSL stream
        info = StreamInfo(
            name=marker_stream_name,
            type="Markers",
            channel_count=1,
            nominal_srate=0,
            channel_format="string",
            source_id=marker_stream_name,
        )
        self.outlet = StreamOutlet(info)

        self.last_selected_key_idx: int | None = None
        self.key_map: dict[int, str] = {}
        self.highlights: dict = {}
        self.decoder_sw = None

        # Set up variables for text to speech, autocompletion, and shifting keyboard layout
        self.sample_idx = 0  # used to iterate through sample symbols, iterated in handle_decoding_event

        self.sample_symbols = []
    
    def _keyfile(self, bank_dir: Path, file_key: str, color:str) -> Path:
        """
        Make the file path for a key image based on the key name and color.

        Parameters
        ----------
        bank_dir : Path
            The directory containing the key image files.
        file_key : str
            The identifier for the key. Special cases such as punctuation (e.g., "comma", "question")
            are mapped to their corresponding filenames.
        color : str
            The color variant of the key image.

        Returns
        -------
        Path
            The full path to the key image file in the format "<bank_dir>/<key>_<color>.png".
        """
        special = {
            "dot": "dot",
            "exclam": "exclam",
            "comma": "comma",
            "question": "question",
            "slash": "slash",
            "backspace": "backspace",
        }
        stem = special.get(file_key, file_key)
        return bank_dir / f"{stem}_{color}.png"

    def add_key(
        self,
        name: str,
        images: list,
        images_lower: list,
        size: tuple[int, int],
        pos: tuple[int, int],
    ) -> None:
        """
        Add a key to the speller.

        Parameters
        ----------
        name: str
            The name of the key.
        images: list
            The list of images associated to the key. Note, index -1 fused for presenting feedback, and index -2 is
            used for cueing.
        images_lower: list
            The list of images associated to the key when the shift key is pressed. If empty, the images list is used.
            The same indices apply as for the images list.
        size: tuple[int, int]
            The size of the key in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the key in pixels provided as (x, y).
        """
        assert name not in self.keys, "Adding a key with a name that already exists!"
        self.keys[name] = []
        for image in images:
            self.keys[name].append(
                visual.ImageStim(
                    win=self.window,
                    name=name,
                    image=image,
                    units="pix",
                    pos=pos,
                    size=size,
                    autoLog=False,
                )
            )
        # Set autoDraw to True for first default key to keep app visible
        self.keys[name][0].setAutoDraw(True)

    def add_text_field(
        self,
        name: str,
        text: str,
        size: tuple[int, int],
        pos: tuple[int, int],
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        text_color: tuple[float, float, float] = (-1.0, -1.0, -1.0),
        alignment: str = "left",
        bold: bool = False,
        anchor: str = "center", 
    ) -> None:
        """
        Add a text field to the speller.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text to display on the text field.
        size: tuple[int, int]
            The size of the text field in pixels provided as (width, height).
        pos: tuple[int, int]
            The position of the text field in pixels provided as (x, y).
        background_color: tuple[float, float, float] (default: (0., 0., 0.))
            The background color of the text field  specified as list of RGB values.
        text_color: tuple[float, float, float] (default: (-1., -1., -1.))
            The text color of the text field  specified as list of RGB values.
        alignment: str (default: "left")
            The alignment of the text in the text field.
        bold: bool (default: False)
            Whether the text is boldface.
        """
        assert name not in self.text_fields, (
            "Adding a text field with a name that already exists!"
        )
        self.text_fields[name] = visual.TextBox2(
            win=self.window,
            name=name,
            text=text,
            units="pix",
            pos=pos,
            size=size,
            letterHeight=0.8 * size[1],
            fillColor=background_color,
            color=text_color,
            alignment=alignment,
            autoDraw=True,
            autoLog=False,
            bold=bold,
            anchor=anchor
        )

    def connect_to_decoder_lsl_stream(self) -> None:
        """
        Connect the decoder to the lsl stream. 
        """
        name = self.cfg["streams"]["decoder_stream_name"]
        speller_logger.info(f'Connecting to decoder stream "{name}".')
        self.decoder_sw = StreamWatcher(name=name)
        self.decoder_sw.connect_to_stream()

    def get_pixels_per_degree(
        self,
    ) -> float:
        """
        Get the pixels per degree of the screen.

        Returns
        -------
        ppd: float
            Pixels per degree of the screen.
        """
        return misc.deg2pix(degrees=1.0, monitor=self.monitor)

    def get_text_field(
        self,
        name: str,
    ) -> str:
        """
        Get the text of the text field.

        Parameters
        ----------
        name: str
            The name of the text field.

        Returns
        -------
        text: str
            The text of the text field.
        """
        assert name in self.text_fields, (
            "Getting text of a text field with a name that does not exists!"
        )
        return self.text_fields[name].text

    def set_text_field(
        self,
        name: str,
        text: str,
    ) -> None:
        """
        Set the text of a text field.

        Parameters
        ----------
        name: str
            The name of the text field.
        text: str
            The text of the text field.
        """
        assert name in self.text_fields, (
            "Setting text of a text field with a name that does not exists!"
        )
        self.text_fields[name].setText(text)
        self.window.flip()

    def log(
        self,
        marker: str,
        on_flip: bool = False,
    ) -> None:
        """
        Log a marker to the marker stream.

        Parameters
        ----------
        marker: str
            The marker to log.
        on_flip: bool (default: False)
            Whether to log on the next frame flip.
        """
        if on_flip:
            self.window.callOnFlip(self.outlet.push_sample, [marker])
        else:
            self.outlet.push_sample([marker])

    def run(
        self,
        sequences: dict,
        duration: float = None,
        start_marker: str = None,
        stop_marker: str = None,
    ) -> None:
        """
        Run a stimulation phase of the speller, which makes the keys flash according to specific sequences.

        Parameters
        ----------
        sequences: dict
            A dictionary containing the stimulus sequences per key.
        duration: float (default: None)
            The duration of the stimulation in seconds. If None, the duration of the first key in the dictionary is
            used.
        start_marker: str (default: None)
            The marker to log when stimulation starts. If None, no marker is logged.
        stop_marker: str (default: None)
            The marker to log when stimulation stops. If None, no marker is logged.
        """

        # Set number of frames
        if duration is None:
            n_frames = len(sequences[list(sequences.keys())[0]])
        else:
            n_frames = int(duration * self.refresh_rate)

        # Set autoDraw to False for full control
        for key in self.keys.values():
            key[0].setAutoDraw(False)

        # Send start marker
        if start_marker is not None:
            self.log(start_marker, on_flip=True)

        # Loop frame flips
        for i in range(n_frames):
            stime = time.time()

            # Check quiting
            if i % 60 == 0:
                if len(event.getKeys(keyList=self.quit_controls)) > 0:
                    self.quit()
                    break

            # Check selection marker
            if self.decoder_sw is not None:
                if self.has_decoding_event():
                    self.handle_decoding_event()
                    break

            # Present keys with color depending on code state
            for name, code in sequences.items():
                self.keys[name][code[i % len(code)]].draw()

            # Check if frame flip can happen within a frame
            etime = time.time() - stime
            if etime >= 1 / self.refresh_rate:
                speller_logger.warn(f"Frame flip took too long ({etime:.6f}), dropping frames!")

            self.window.flip()
        else:
            speller_logger.debug(f"All {n_frames=} shown.")

        # Send stop marker
        if stop_marker is not None:
            self.log(stop_marker)

        # Set autoDraw to True to keep speller visible
        for key in self.keys.values():
            key[0].setAutoDraw(True)

    def quit(
        self,
    ) -> None:
        """
        Quit the speller, close the window.
        """
        for key in self.keys.values():
            key[0].setAutoDraw(True)

        if self.window is not None:
            self.window.flip()
            self.window.setMouseVisible(True)
            self.window.close()

    def has_decoding_event(self) -> bool:
        """
        Check if the LSL stream contained a `speller_select <key_idx>` marker.

        Returns
        -------
        bool:
            True if there is a decoding event, False otherwise.
        """
        # disregard all previous data
        self.decoder_sw.n_new = 0
        self.decoder_sw.update()

        if self.decoder_sw.n_new != 0:
            prediction = self.decoder_sw.unfold_buffer()[
                -self.decoder_sw.n_new :
            ].flatten()

            speller_logger.debug(f"Received: prediction={prediction}")
            selections = prediction[prediction >= 0]

            if len(selections) > 0:
                self.last_selected_key_idx = int(selections[-1])
                speller_logger.debug(f"Selection: {self.last_selected_key_idx}, {selections=}")
                return True

        return False

    def handle_decoding_event(self) -> None:
        """
        Handle a decoding event by updating the spelled text and presenting feedback.

        The predicted key is determined from the latest decoder output, applied to the
        text field, and visually highlighted to provide feedback.
        """
        # Decoding
        speller_logger.info("Waiting for decoding")

        # with fewer keys, it is possible for random value to be out of bounds so take idx % number of keys
        prediction = self.last_selected_key_idx % len(self.key_map)
        prediction_key = self.key_map[prediction]

        speller_logger.debug(
            f"Decoding: prediction={prediction} prediction_key={prediction_key}"
        )

        # Spelling
        text = self.get_text_field("text")

        # if there is an available list of sample symbols, use them, otherwise use the random prediction
        if self.sample_idx < len(self.sample_symbols):
            symbol = self.sample_symbols[self.sample_idx]
            if (
                symbol in KEY_MAPPING.values()
            ):  # update the prediction key to the recognized key name if the symbol is
                # a special character (i.e. / -> slash)
                prediction_key = list(KEY_MAPPING.keys())[
                    list(KEY_MAPPING.values()).index(symbol)
                ]
            elif symbol == " ":
                prediction_key = "space"
            else:
                prediction_key = symbol
            self.sample_idx += 1  # increment the sample index to move to the next symbol for the next decode event
        else:
            symbol = prediction_key

        # if the symbol is a key in KEY_MAPPING, change it to a symbol to be added to the text field
        # (i.e. "slash" -> "/")
        if symbol in KEY_MAPPING:
            symbol = KEY_MAPPING[symbol]

        # Handle special keys
        if symbol == self.cfg["speller"]["key_space"]:
            # Add a whitespace
            text = text + " "
        elif symbol == self.cfg["speller"]["key_clear"]:
            # Clear the full sentence
            text = ""
        elif symbol == self.cfg["speller"]["key_backspace"]:
            # Perform a backspace
            text = text[:-1]
        else:
            text += symbol

        # update the text field with the new text
        self.set_text_field(name="text", text=text)
        speller_logger.debug(f"Feedback: symbol={symbol} text={text}")

        # Feedback
        speller_logger.info(f"Presenting feedback {prediction_key} ({prediction})")
        self.highlights[prediction_key] = [-1]
        self.run(
            sequences=self.highlights,
            duration=self.cfg["speller"]["timing"]["feedback_s"],
            start_marker=f"{self.cfg['speller']['markers']['feedback_start']};label={prediction};key={prediction_key}",
            stop_marker=self.cfg["speller"]["markers"]["feedback_stop"],
        )

        # remove the highlight from the selected key
        self.highlights[prediction_key] = [0]

    def init_highlights_with_zero(self) -> None:
        """
        Initialize the highlight state for all speller keys.
        """
        # Setup highlights
        self.highlights = dict()
        keys_from_cfg = self.cfg["speller"]["keys"]["keys_upper"]

        for row in keys_from_cfg:
            for key in row:
                self.highlights[key] = [0]

    def load_bank_images(self, condition: str, contrast: int) -> None:
        """
        Load and assign key images for a specific speller condition and contrast level.

        Parameters
        ----------
        condition : str
            The name of the image bank to load. 
        contrast : int
            The contrast level to use for the image bank.

        Raises
        ------
        ValueError
            If the image bank does not define at least two colors in its configuration.
        """
        images_dir = Path(self.cfg["speller"]["images_dir"])
        bank = _get_bank(self.cfg, condition)
        
        colors = bank["colors"]
        if len(colors) < 2:
            raise ValueError(f"Bank '{condition}' must define at least [ON, OFF] in 'colors'")

        on_color, off_color = colors[0], colors[1]
        cue_color = colors[2] if len(colors) > 2 else "green"
        feedback_color = colors[3] if len(colors)>2 else "blue"

        bank_dir = images_dir / condition / f"contrast-{int(contrast)}"

        for name, stims in self.keys.items():
            stims[0].image = self._keyfile(bank_dir, name, off_color)
            stims[1].image = self._keyfile(bank_dir, name, on_color)
            stims[2].image = self._keyfile(bank_dir, name, cue_color)
            stims[3].image = self._keyfile(bank_dir, name, feedback_color)


def build_timeline(cfg: dict, code_to_key: dict) -> list[dict]:
    """
    Build a timeline where every (condition, contrast) cell
    is presented exactly once for every key.

    Randomization is preserved by shuffling trials and then
    splitting them into blocks.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary containing speller, decoder, and stimulus settings.
    code_to_key : dict
        Dictionary mapping code indices to speller keys, used to determine the set of keys
        included in the timeline.

    Returns
    -------
    list[dict]
        list of dicts with keys: condition, contrast, key_idx
    """

    rng = random.Random(cfg.get("speller", {}).get("random_seed"))

    # Collect all condition × contrast cells from the config 
    banks = cfg["speller"]["banks"]
    cells = []
    for bank in banks:
        cond = bank["condition"]
        for cont in bank["contrasts"]:
            cells.append((cond, int(cont)))

    if not cells:
        raise ValueError("No condition×contrast cells found in cfg['speller']['banks'].")

    # All key indices used in the speller 
    key_indices = sorted(code_to_key.keys())
    if not key_indices:
        raise ValueError("code_to_key is empty.")

    n_cells = len(cells)
    n_keys = len(key_indices)
    total_trials = n_cells * n_keys

    # Build exactly one trial per (cell, key) 
    trials = []
    for cond, cont in cells:
        for k in key_indices:
            trials.append({
                "condition": cond,
                "contrast": cont,
                "key_idx": k,
            })

    assert len(trials) == total_trials

    # Shuffle all trials
    rng.shuffle(trials)

    # Split into blocks
    block_size = int(cfg["training"]["block_size"])
    if block_size <= 0:
        raise ValueError("cfg['training']['block_size'] must be > 0")

    timeline = []
    for start in range(0, total_trials, block_size):
        block = trials[start:start + block_size]
        # Shuffle inside each block as well
        rng.shuffle(block)
        timeline.extend(block)

    # sanity check
    assert len(timeline) == total_trials, (
        f"Built {len(timeline)} trials, expected {total_trials}"
    )

    return timeline

def setup_speller(cfg: dict) -> Speller:
    """
    Initialize and configure the visual c-VEP speller interface.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary containing screen, speller, stimulus, and control
        settings.

    Returns
    -------
    Speller
        Configured speller instance used for stimulus presentation.
    """
    # Setup speller
    speller = Speller(
        screen_resolution=cfg["speller"]["screen"]["resolution"],
        width_cm=cfg["speller"]["screen"]["width_cm"],
        distance_cm=cfg["speller"]["screen"]["distance_cm"],
        refresh_rate=cfg["speller"]["screen"]["refresh_rate_hz"],
        screen_id=cfg["speller"]["screen"]["id"],
        full_screen=cfg["speller"]["screen"]["full_screen"],
        background_color=cfg["speller"]["screen"]["background_color"],
        marker_stream_name=cfg["streams"]["marker_stream_name"],
        quit_controls=cfg["speller"]["controls"]["quit"],
        cfg=cfg,
    )
    win_w, win_h = map(int, speller.window.size)
    ppd = speller.get_pixels_per_degree()

    x_pos = 0
    x_size = win_w
    y_pos = int(win_h / 2 - (cfg["speller"]["text_fields"]["height_dva"] * ppd) / 2)
    y_size = int(cfg["speller"]["text_fields"]["height_dva"] * ppd)

    speller.add_text_field(
     name="text",
        text="",
        size=(x_size, y_size),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
        text_color=(-1.0, -1.0, -1.0),
    )

    x_pos = 0
#    add a small visual-angle margin so it never clips
    bottom_margin_px = 40
    y_pos = -win_h / 2+ bottom_margin_px

    speller.add_text_field(
        name="messages",
        text="",
        size=(win_w, int(cfg["speller"]["text_fields"]["height_dva"] * ppd)),
        pos=(x_pos, y_pos),
        background_color=cfg["speller"]["text_fields"]["background_color"],
        alignment="center",
        anchor="bottom-center"
    )

    # Add keys
    keys_from_cfg = cfg["speller"]["keys"]["keys_upper"]
    for y in range(len(keys_from_cfg)):
        for x in range(len(keys_from_cfg[y])):
            x_pos = int(
                (x - len(keys_from_cfg[y]) / 2 + 0.5)
                * (
                    cfg["speller"]["keys"]["width_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
            )
            y_pos = int(
                -(y - len(keys_from_cfg) / 2)
                * (
                    cfg["speller"]["keys"]["height_dva"]
                    + cfg["speller"]["keys"]["space_dva"]
                )
                * ppd
                - cfg["speller"]["text_fields"]["height_dva"] * ppd
            )

            filename_key_map = {
                "dot": "dot", "exclam": "exclam", "comma": "comma", "question": "question",
                "slash": "slash", "backspace": "backspace", "space": "space"
            }

            base = Path(cfg["speller"]["images_dir"]) / "classic" / "contrast-100"
            key_name = keys_from_cfg[y][x]
            file_key = filename_key_map.get(key_name, key_name)


            images = [
                base / f"{file_key}_black.png",
                base / f"{file_key}_white.png",
                base / f"{file_key}_green.png",
                base / f"{file_key}_blue.png"
            ]

            # Create a border to make the keys more visible
            border = visual.Rect(
                win=speller.window,
                pos=(x_pos, y_pos),
                width=int(cfg["speller"]["keys"]["width_dva"] * ppd) + 2,
                height=int(cfg["speller"]["keys"]["height_dva"] * ppd) + 2,
                lineColor=(0.5, 0.5, 0.5),
                fillColor=None,
                lineWidth=2,
                autoLog=False
            )
            border.setAutoDraw(True)

            speller.add_key(
                name=keys_from_cfg[y][x],
                images=images,
                images_lower=[],
                size=(
                    int(cfg["speller"]["keys"]["width_dva"] * ppd),
                    int(cfg["speller"]["keys"]["height_dva"] * ppd),
                ),
                pos=(x_pos, y_pos),
            )
    
    # Make sure the messages bar is visible
    speller.text_fields["messages"].setAutoDraw(False)
    speller.text_fields["messages"].setAutoDraw(True)
    speller.text_fields["text"].setAutoDraw(False)
    speller.text_fields["text"].setAutoDraw(True)

    speller.init_highlights_with_zero()

    return speller


def create_key2seq_and_code2key(cfg: dict, phase: str) -> tuple[dict, dict]:
    """
    Create mappings between speller keys and stimulus code sequences.

    Parameters
    ----------
    cfg: dict
        Configuration dictionary containing speller, decoder, and stimulus settings.
    phase : str
        Experimental phase for which the mappings are created (e.g., "training" or "online").

    Returns
    -------
    key_to_sequence : dict
        Dictionary mapping each speller key to its corresponding stimulus sequence.
    code_to_key : dict
        Dictionary mapping code indices to speller keys.
    """
    codes_file = Path(cfg[phase]["codes_file"])

    # Setup code sequences from the correct phase
    codes = np.load(Path(cfg["speller"]["codes_dir"]) / codes_file)["codes"]
    codes = np.repeat(
        codes,
        int(
            cfg["speller"]["screen"]["refresh_rate_hz"]
            / cfg["speller"]["presentation_rate_hz"]
        ),
        axis=1,
    )

    # Optimal layout and subset:
    subset_layout_file = cfg["decoder"]["decoder_subset_layout_file"]
    if phase == "online" and len(subset_layout_file) > 0:
        # Fetch the subset and layout file location
        if os.path.isfile(subset_layout_file):
            with open(subset_layout_file, "r") as infile:
                data = json.load(infile)
                subset = np.array(data["subset"])
                layout = np.array(data["layout"])

            # Extra assertion check. The speller and decoder online/training code files should match.
            assert codes_file.name == data["codes_file"], (
                "The stimuli of the speller and decoder are not the same, please check."
            )

            # Set the loaded codes with subset and optimal layout
            # Note that this means that while i_code still refers to indices 0 trough to n_keys
            # The actual code that's placed there might for example originally be indices 59, 12, 0...
            # You can find the actual index values of the original code file by printing/comparing the optimal_layout
            # np array.
            codes = codes[subset, :]
            codes = codes[layout, :]
            speller_logger.info("Stimulus subset and layout applied.")
        else:
            speller_logger.info(f"Subset and layout file {subset_layout_file} not found.")
    else:
        speller_logger.debug("No stimulus subset or layout applied.")

    key_to_sequence = dict()
    code_to_key = dict()
    i_code = 0
    keys_from_cfg = cfg["speller"]["keys"]["keys_upper"]
    for row in keys_from_cfg:
        for key in row:
            key_to_sequence[key] = codes[i_code, :].tolist()
            code_to_key[i_code] = key
            i_code += 1

    return key_to_sequence, code_to_key


def run_speller_paradigm(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
) -> int:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.

    Returns
    -------
    flag: int
        Whether the process ran without errors or with.
    """
    cfg = toml.load(config_path)

    speller = setup_speller(cfg)

    if phase != "training":
        speller.connect_to_decoder_lsl_stream()

    key_to_sequence, code_to_key = create_key2seq_and_code2key(cfg, phase)
    speller.key_map = code_to_key
    n_classes = len(code_to_key)

    # Build a shuffled timeline
    timeline = build_timeline(cfg, code_to_key)

    # Wait to start run
    speller_logger.info("Waiting for button press to start")
    speller.set_text_field(name="messages", text="Press button to start.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Log info
    python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    speller.log(
        marker=f"version;python={python_version};psychopy={psychopy.__version__}"
    )
    speller.log(marker=f"setup;codes={key_to_sequence};labels={code_to_key}")

    # Start run
    speller_logger.info("Starting")
    speller.log(marker="start_run")
    speller.set_text_field(name="messages", text="Starting...")
    speller.run(sequences=speller.highlights, duration=5.0)
    speller.text_fields["messages"].setAutoDraw(True)
    speller.set_text_field(name="messages", text="")

    block_size = cfg["training"].get("block_size", 40)

    # Loop trials
    n_trials = len(timeline)
    for i_trial in range(n_trials):
        speller_logger.info(f"Initiating trial {1 + i_trial}/{n_trials}")

        if phase == "training":
            row = timeline[i_trial]
            condition = row["condition"]
            contrast = row["contrast"]
            target = row["key_idx"]
            target_key = code_to_key[target]

            speller.load_bank_images(condition, contrast)

            # Cue
            speller_logger.info(f"Cueing {target_key} ({target})")
            speller.highlights[target_key] = [-2]
            speller.run(
                sequences=speller.highlights,
                duration=cfg["speller"]["timing"]["cue_s"],
                start_marker=f"{cfg['speller']['markers']['cue_start']};key_idx={target};key={target_key};condition={condition};contrast={contrast}",
                stop_marker=cfg["speller"]["markers"]["cue_stop"],
            )
            speller.highlights[target_key] = [0]

            speller.text_fields["messages"].setAutoDraw(True)

            # Trial
            speller_logger.info("Starting stimulation")
            speller.run(
                sequences=key_to_sequence,
                duration=cfg["speller"]["timing"]["trial_s"],
                start_marker=f"{cfg['speller']['markers']['trial_start']};condition={condition};contrast={contrast}",
                stop_marker=cfg["speller"]["markers"]["trial_stop"],
            )
            speller.text_fields["messages"].setAutoDraw(True)
            
            is_end_block = ((i_trial + 1) % block_size == 0)
            is_not_last_trial = (i_trial + 1) < n_trials
            
            if is_end_block and is_not_last_trial:

                print(f"BREAK START: block {(i_trial + 1) // block_size}")

                speller.log("start_rest")
                speller_logger.info("Waiting for button press to continue")
                speller.set_text_field(name="messages", text="Press button to continue.")
                event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
                speller.log("stop_rest")
                speller.set_text_field(name="messages", text="")
                
                speller_logger.info("Starting")
                speller.set_text_field(name="messages", text="Starting...")
                speller.run(sequences=speller.highlights, duration=5.0)
                speller.text_fields["messages"].setAutoDraw(True)
                speller.set_text_field(name="messages", text="")

        if phase == "online" and speller.get_text_field("text").endswith(
            cfg["speller"]["quit_phrase"]
        ):
            break

    # Wait to stop
    speller_logger.info("Waiting for button press to stop")
    speller.set_text_field(name="messages", text="Press button to stop.")
    event.waitKeys(keyList=cfg["speller"]["controls"]["continue"])
    speller.set_text_field(name="messages", text="")

    # Stop run
    speller_logger.info("Stopping")
    speller.log(marker="stop_run")
    speller.set_text_field(name="messages", text="Stopping...")
    speller.run(sequences=speller.highlights, duration=5.0)
    speller.set_text_field(name="messages", text="")
    speller.quit()

    return 0


# make this the cli starting point
def cli_run(
    phase: str = "training",
    config_path: Path = Path("./configs/speller.toml"),  # relative to the project root
    log_level: int = 30,
) -> None:
    """
    Run the speller in a particular phase (training or online).

    Parameters
    ----------
    phase: str (default: "training")
        The phase of the speller being either training or online. During training, the user is cued to attend to a
        random target key every trail. During online, the user attends to their target, while their EEG is decoded and
        the decoded key is used to perform an action (e.g., add a symbol to a sentence, backspace, etc.). In the online
        phase, the speller will continuously query an LSL marker stream to look for a potential decoding result from
        the decoder module. If the stream contains a marker `speller_select <key_idx>`, the speller will
        stop the presentation and will show the selected symbol.
    config_path: Path (default: "./configs/speller.toml")
        The path to the configuration file containing session specific hyperparameters for the speller setup.
    log_level : int (default: 30)
        The logging level to use.
    """
    # activate the console logging if started via CLI
    logger = get_logger("cvep-speller", add_console_handler=True)
    logger.setLevel(log_level)

    run_speller_paradigm(phase=phase, config_path=config_path)

if __name__ == "__main__":
    Fire(cli_run)