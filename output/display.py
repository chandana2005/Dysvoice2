"""
display.py - Transcript Display Module
DysVoice | Person 3 - Output & Integration

Shows the transcribed text on screen. Currently prints to the terminal
as a fallback. When the Raspberry Pi and OLED hardware arrive, swap the
terminal print for the OLED library call (see the TODO comment below).
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── OLED flag ────────────────────────────────────────────────────────────────
# Set this to True only when running on the Raspberry Pi with the OLED
# connected. Keeps the same display_text() call working in both environments.
OLED_ENABLED = False

if OLED_ENABLED:
    # These imports only work on the Pi with the hardware connected.
    # Leave commented out until hardware arrives.
    # from luma.core.interface.serial import i2c
    # from luma.oled.device import ssd1306
    # from luma.core.render import canvas
    # from PIL import ImageFont
    pass


def display_text(text: str) -> None:
    """
    Display the transcript text on screen.

    In software-only mode (OLED_ENABLED = False), prints a clearly
    formatted transcript line to the terminal so the panel can read it.

    When hardware is available, set OLED_ENABLED = True and uncomment
    the OLED rendering block inside this function to push text to the
    physical screen instead.

    Args:
        text (str): The transcribed text string to display.

    Returns:
        None
    """
    if not text or not text.strip():
        return

    if OLED_ENABLED:
        # TODO: swap this block in when Pi + OLED are connected.
        # serial = i2c(port=1, address=0x3C)
        # device = ssd1306(serial)
        # with canvas(device) as draw:
        #     draw.text((0, 0), text, fill="white")
        pass
    else:
        # Terminal fallback — used during all software development
        _print_terminal(text)


def _print_terminal(text: str) -> None:
    """
    Print the transcript to the terminal in a clear, readable format.

    Args:
        text (str): The transcribed text to display.
    """
    width = 50
    print("┌" + "─" * width + "┐")
    print(f"│  Transcript: {text:<{width - 15}}│")
    print("└" + "─" * width + "┘")


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== display.py test ===\n")

    test_phrases = [
        "please bring me water",
        "turn off the lights",
        "I need help",
        "could not understand, please repeat",
    ]

    for phrase in test_phrases:
        display_text(phrase)

    print("\n=== display.py tests complete ===")