import os
import random
from pathlib import Path
import toml
from psychopy import event
from cvep_speller.speller import setup_speller, create_key2seq_and_code2key

def run_questionnaire_viewer(config_path: Path | None = None) -> None:
    """
    Run an interactive stimulus viewer for subjective questionnaire assessment.

    This function presents all stimulus conditions and contrast levels in a randomized
    order, allowing participants to view each condition and repeat presentations if
    desired.

    Parameters
    ----------
    config_path : Path or None (default: None)
        Path to the configuration file. If None, the default speller configuration
        is loaded.
    """
    base_dir = Path(__file__).resolve().parent

    os.chdir(base_dir)

    if config_path is None:
        config_path = base_dir / "configs" / "speller.toml"

    print(f"Using config file: {config_path}")
    cfg = toml.load(config_path)

    speller = setup_speller(cfg)
    key_to_sequence, code_to_key = create_key2seq_and_code2key(cfg, phase="training")

    banks = cfg["speller"]["banks"]
    all_conditions: list[tuple[str, int]] = []
    for bank in banks:
        cond = bank["condition"]
        for cont in bank["contrasts"]:
            all_conditions.append((cond, int(cont)))

    random.shuffle(all_conditions)

    print("\n=== QUESTIONNAIRE PRESENTATION ORDER ===")
    for i, (cond, cont) in enumerate(all_conditions, start=1):
        print(f"{i:02d}: condition={cond}, contrast={cont}")
    print("========================================\n")

    speller.set_text_field("messages", text="Press C to show conditions.")
    event.waitKeys(keyList=["c"])
    speller.set_text_field("messages", text="")

    for cond, cont in all_conditions:
        while True:
            print(f"Showing condition={cond}, contrast={cont}")
            speller.load_bank_images(cond, cont)

            speller.set_text_field("messages", text="")

            speller.run(
                sequences=key_to_sequence,
                duration=3.0,
                start_marker=f"questionnaire_trial_start;condition={cond};contrast={cont}",
                stop_marker="questionnaire_trial_stop",
            )

            speller.set_text_field(
                "messages",
                text="Press C (next) or R (repeat).",
            )

            key = event.waitKeys(keyList=["c", "r"])[0]
            speller.set_text_field("messages", text="")

            if key == "c":
                break
            elif key == "r":
                continue

    speller.set_text_field(
        "messages",
        text="Questionnaire run finished. Press key to quit.",
    )
    event.waitKeys()
    speller.quit()

if __name__ == "__main__":
    run_questionnaire_viewer()
