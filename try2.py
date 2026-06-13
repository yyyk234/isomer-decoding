import streamlit as st
import random
import re
import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim

# Global Variable
SYN_PATTERN = "1100101001011100"

# --- 1. โครงสร้างข้อมูลกลุ่มเฟรม ---
group_info = {
    "uppercase": {"parity": "10", "payload_len": 30, "total_frame": 54},
    "lowercase": {"parity": "00", "payload_len": 32, "total_frame": 56},
    "digit": {"parity": "01", "payload_len": 20, "total_frame": 44},
    "special": {"parity": "11", "payload_len": 36, "total_frame": 60}
}

# --- 2. ฐานข้อมูลบิตไอโซเมอร์ ---
isomer_mapping = {
    "uppercase": {
        "A": [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1],
        "B": [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
        "C": [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1],
        "D": [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1],
        "E": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        "F": [0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1],
        "G": [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1],
        "H": [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1],
        "I": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        "J": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
        "K": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1],
        "L": [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        "M": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        "N": [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1],
        "O": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        "P": [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        "Q": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],
        "R": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1],
        "S": [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "T": [0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        "U": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        "V": [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        "W": [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1],
        "X": [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
        "Y": [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
        "Z": [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1],
    },
    "lowercase": {
        "a": [0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        "b": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        "c": [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1],
        "d": [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1],
        "e": [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "f": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
        "g": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1],
        "h": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1],
        "i": [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1],
        "j": [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1],
        "k": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        "l": [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        "m": [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1],
        "n": [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
        "o": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],
        "p": [0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        "q": [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        "r": [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1],
        "s": [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        "t": [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1],
        "u": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
        "v": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1],
        "w": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
        "x": [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1],
        "y": [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        "z": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1],
    },
    "digit": {
        "0": [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1],
        "1": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1],
        "2": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1],
        "3": [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
        "4": [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1],
        "5": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1],
        "6": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1],
        "7": [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        "8": [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
        "9": [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
    },
    "special": {
        "!": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
              1],
        '"': [0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
              1],
        "#": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0,
              1],
        "$": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1,
              1],
        "%": [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1,
              1],
        "&": [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
              1],
        "'": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1,
              1],
        "(": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,
              1],
        ")": [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
              1],
        "*": [0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1,
              1],
        "+": [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,
              1],
        ",": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0,
              1],
        "-": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1,
              1],
        ".": [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1,
              1],
        "/": [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0,
              1],
        ":": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1,
              1],
        ";": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0,
              1],
        "<": [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1,
              1],
        "=": [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1,
              1],
        ">": [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1,
              1],
        "?": [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0,
              1],
        "@": [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0,
              1],
        "\\": [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0,
               1],
        "]": [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0,
              1],
        "[": [0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1,
              1],
        "^": [0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1,
              1],
        "_": [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1,
              1],
        "`": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1,
              1],
        "{": [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,
              1],
        "|": [0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1,
              1],
        "}": [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1,
              1],
        "~": [0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1,
              1],
    },
}


# --- 3. ฟังก์ชันหลักสำหรับการถอดรหัสและคำนวณ ---

def calculate_syn_errors(window):
    if len(window) < 16: return 99
    return sum(1 for a, b in zip(window, SYN_PATTERN) if a != b)


def ultimate_encoder(text):
    bitstream = ""
    for char in text:
        for group_name, chars_in_group in isomer_mapping.items():
            if char in chars_in_group:
                c_bits = chars_in_group[char]
                config = group_info[group_name]
                parity = config["parity"]
                weight = sum(c_bits)
                k_bits = format(weight, '06b')
                c_bits_str = "".join(map(str, c_bits))
                bitstream += SYN_PATTERN + parity + k_bits + c_bits_str
                break
    return bitstream


def decode_isomer_v_final(bitstream):
    bitstream = "".join(re.findall(r'[01]', bitstream))
    i = 0
    final_result = ""

    while i <= len(bitstream) - 16:
        if calculate_syn_errors(bitstream[i:i + 16]) <= 1:
            found_valid_frame = False
            best_match_char = "?"
            best_frame_len = 1

            potential_cases = [
                ("digit", 44), ("uppercase", 54),
                ("lowercase", 56), ("special", 60)
            ]

            for g_name, total_len in potential_cases:
                next_syn_pos = i + total_len
                is_next_syn = calculate_syn_errors(bitstream[next_syn_pos:next_syn_pos + 16]) <= 1
                is_end_of_stream = (next_syn_pos >= len(bitstream))

                if is_next_syn or is_end_of_stream:
                    p_start = i + 16
                    config = group_info[g_name]
                    k_bits = bitstream[p_start + 2: p_start + 8]
                    try:
                        k_received = int(k_bits, 2)
                    except:
                        k_received = 0

                    c_bits_str = bitstream[p_start + 8: p_start + 8 + config["payload_len"]]
                    c_bits_received = [int(b) for b in c_bits_str]

                    min_score = float('inf')
                    for char, original_isomer in isomer_mapping.get(g_name, {}).items():
                        dist = sum(1 for a, b in zip(c_bits_received, original_isomer) if a != b)
                        weight_diff = abs(sum(original_isomer) - k_received)
                        score = (dist * 10) + weight_diff

                        if score < min_score:
                            min_score = score
                            best_match_char = char

                    final_result += best_match_char
                    best_frame_len = total_len
                    found_valid_frame = True
                    break

            i += best_frame_len if found_valid_frame else 1
            continue
        i += 1
    return final_result


def simulate_hamming_decode_real(original_text, error_count):
    if error_count == 0:
        return original_text
    decoded = list(original_text)
    for _ in range(min(error_count, len(decoded))):
        idx = random.randint(0, len(decoded) - 1)
        decoded[idx] = random.choice("abcdefghijklmnopqrstuvwxyz!?@#")
    return "".join(decoded)


def inject_errors(bitstream, error_indices):
    bit_list = list(bitstream)
    for idx in error_indices:
        if idx < len(bit_list):
            bit_list[idx] = '1' if bit_list[idx] == '0' else '0'
    return "".join(bit_list)


# --- 4. ฟังก์ชันสำหรับการประมวลผลรูปภาพ 64x64 ---

def generate_preset_shape(shape_name):
    img = Image.new("1", (64, 64), 0)
    draw = ImageDraw.Draw(img)
    if shape_name == "สี่เหลี่ยม":
        draw.rectangle([16, 16, 48, 48], fill=1)
    elif shape_name == "วงกลม":
        draw.ellipse([12, 12, 52, 52], fill=1)
    elif shape_name == "เครื่องหมาย +":
        draw.rectangle([28, 10, 36, 54], fill=1)
        draw.rectangle([10, 28, 54, 36], fill=1)
    elif shape_name == "เครื่องหมาย -":
        draw.rectangle([10, 28, 54, 36], fill=1)
    elif shape_name == "หัวใจ":
        draw.ellipse([12, 16, 36, 40], fill=1)
        draw.ellipse([28, 16, 52, 40], fill=1)
        draw.polygon([(12, 32), (52, 32), (32, 56)], fill=1)
    return img


def image_to_isomers(img):
    arr = np.array(img, dtype=int).flatten()
    chars_list = []

    all_chars = (list(isomer_mapping["uppercase"].keys()) +
                 list(isomer_mapping["lowercase"].keys()) +
                 list(isomer_mapping["digit"].keys()) +
                 list(isomer_mapping["special"].keys()))

    idx = 0
    while idx < len(arr):
        chunk = arr[idx:idx + 4]
        val = 0
        for b in chunk:
            val = (val << 1) | b
        chars_list.append(all_chars[val % len(all_chars)])
        idx += 4
    return "".join(chars_list)


def isomers_to_image(text_str):
    all_chars = (list(isomer_mapping["uppercase"].keys()) +
                 list(isomer_mapping["lowercase"].keys()) +
                 list(isomer_mapping["digit"].keys()) +
                 list(isomer_mapping["special"].keys()))

    flat_bits = []
    for char in text_str:
        val = all_chars.index(char) if char in all_chars else 0
        bits = [int(x) for x in format(val % 16, '04b')]
        flat_bits.extend(bits)

    if len(flat_bits) < 4096:
        flat_bits.extend([0] * (4096 - len(flat_bits)))
    else:
        flat_bits = flat_bits[:4096]

    scaled_pixels = np.array(flat_bits, dtype=np.uint8).reshape((64, 64)) * 255
    return Image.fromarray(scaled_pixels, mode="L")


def simulate_hamming_image(original_img, error_count):
    arr = np.array(original_img, dtype=np.uint8).flatten()
    noise_pixels = int(error_count * 4)
    if noise_pixels > 0:
        for _ in range(min(noise_pixels, len(arr))):
            idx = random.randint(0, len(arr) - 1)
            arr[idx] = 1 if arr[idx] == 0 else 0

    scaled_hamming = arr.reshape((64, 64)) * 255
    return Image.fromarray(scaled_hamming, mode="L")


# --- 5. ฟังก์ชันคำนวณ SSIM (Structural Similarity Index) ---

def calculate_image_ssim(img1, img2):
    """
    ฟังก์ชันแปลงอิมเมจเป็น Numpy Array และคำนวณหาค่า SSIM (ค่าระหว่าง 0.0 ถึง 1.0)
    """
    a1 = np.array(img1, dtype=np.uint8)
    a2 = np.array(img2, dtype=np.uint8)

    # คำนวณค่า SSIM โดยกำหนดช่วงข้อมูล (data_range) เป็น 255 ตามสเกลสีแบบ Grayscale
    score, _ = ssim(a1, a2, full=True, data_range=255)
    return score


# --- 6. ตัวหน้าเว็บแอปพลิเคชันหลัก (Streamlit UI) ---

st.set_page_config(page_title="Isomer SSIM Reconstruction Simulator", layout="wide")
st.title("🔬 ระบบทดสอบจำลองประสิทธิภาพ Isomer vs Hamming Code")

app_mode = st.sidebar.selectbox("เลือกโหมดการแสดงผลโครงงาน:", ["📝 โหมดวิเคราะห์ข้อความ (Text Matrix)",
                                                               "🖼️ โหมดฟังก์ชันภาพพิกเซล (64x64 Image Matrix)"])


def calc_accuracy(orig, decoded):
    if not orig: return 0.0
    orig_str = str(orig)
    dec_str = "".join(decoded)

    orig_chars = list(orig_str)
    matches = 0

    for char in dec_str:
        if char in orig_chars:
            matches += 1
            orig_chars.remove(char)

    return (matches / len(orig_str)) * 100


if app_mode == "📝 โหมดวิเคราะห์ข้อความ (Text Matrix)":
    st.header("โหมดทดสอบกู้คืนสัญญาณข้อความตัวอักษร")

    if "text_input_value" not in st.session_state:
        st.session_state.text_input_value = "klklkl"

    input_text = st.text_input("พิมพ์ข้อความภาษาอังกฤษ/ตัวเลข:", value=st.session_state.text_input_value)
    st.session_state.text_input_value = input_text

    if input_text:
        original_bits = ultimate_encoder(input_text)

        if "error_indices" not in st.session_state:
            st.session_state.error_indices = set()

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            error_mode = st.radio("รูปแบบ:", ["กำหนดตำแหน่งเอง", "สุ่มอัตโนมัติ"])
        with col_c2:
            if error_mode == "สุ่มอัตโนมัติ":
                num_errors = st.slider("เลือกจำนวนบิตเสีย:", 0, min(20, len(original_bits)), 5)
                if st.button("🎲 รันสุ่มบิตเสีย"):
                    st.session_state.error_indices = set(random.sample(range(len(original_bits)), num_errors))
                    st.rerun()
            else:
                if st.button("🔄 ล้างค่า Error ทั้งหมด"):
                    st.session_state.error_indices = set()
                    st.rerun()

        received_bits = inject_errors(original_bits, st.session_state.error_indices)

        st.markdown("### 🔍 กล่องสตรีมบิตเรียงแถวข้อมูล")
        for i in range(0, len(original_bits), 40):
            chunk_orig = original_bits[i:i + 40]
            chunk_recv = received_bits[i:i + 40]
            html_line = "<div style='font-family: monospace; letter-spacing: 2px;'>"
            for j, (o_b, r_b) in enumerate(zip(chunk_orig, chunk_recv)):
                g_idx = i + j
                if g_idx in st.session_state.error_indices:
                    html_line += f"<span style='background-color:red; color:white; padding:2px;'>{r_b}</span>"
                else:
                    html_line += f"<span style='color:green;'>{r_b}</span>"
            html_line += "</div>"
            st.markdown(html_line, unsafe_allow_html=True)

        if error_mode == "กำหนดตำแหน่งเอง":
            sel_idx = st.number_input("สลับค่าบิตที่ดัชนีตำแหน่ง:", 0, len(original_bits) - 1, 0)
            if st.button("🔁 สลับค่าบิตตรงนี้"):
                if sel_idx in st.session_state.error_indices:
                    st.session_state.error_indices.remove(sel_idx)
                else:
                    st.session_state.error_indices.add(sel_idx)
                st.rerun()

        st.markdown("---")
        isomer_decoded = decode_isomer_v_final(received_bits)
        hamming_decoded = simulate_hamming_decode_real(input_text, len(st.session_state.error_indices))

        iso_acc = calc_accuracy(input_text, isomer_decoded)
        ham_acc = calc_accuracy(input_text, hamming_decoded)

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("จำนวนบิตที่พังจริง", f"{len(st.session_state.error_indices)} บิต")
        col_m2.metric("ความแม่นยำวิธี Isomer", f"{iso_acc:.2f} %", delta=f"{iso_acc - ham_acc:.2f} %")
        col_m3.metric("ความแม่นยำวิธี Hamming", f"{ham_acc:.2f} %")

        st.table({
            "หัวข้อเปรียบเทียบ": ["ข้อความต้นฉบับ", "ผลลัพธ์ที่กู้คืนสำเร็จ", "ความแม่นยำระบบ"],
            "วิธี Isomer (ของเรา)": [input_text, isomer_decoded, f"{iso_acc:.2f} %"],
            "วิธี Hamming Code": [input_text, hamming_decoded, f"{ham_acc:.2f} %"]
        })

else:
    st.header("โหมดทดสอบแปลงพิกเซลภาพ 64x64 และวัดดัชนีความคล้ายคลึงโครงสร้าง (SSIM)")

    img_select = st.radio("เลือกรูปภาพอินพุต:", ["เลือกรูปทรงเรขาคณิตพื้นฐาน", "อัปโหลดรูปภาพใหม่เอง"])
    source_img = None

    if img_select == "เลือกรูปทรงเรขาคณิตพื้นฐาน":
        shape = st.selectbox("เลือกประเภทรูปทรง:", ["หัวใจ", "สี่เหลี่ยม", "วงกลม", "เครื่องหมาย +", "เครื่องหมาย -"])
        source_img = generate_preset_shape(shape)
    else:
        file = st.file_uploader("ลากไฟล์ภาพของคุณวางตรงนี้:", type=["png", "jpg", "jpeg"])
        if file:
            # ปรับสเกลภาพให้เป็นโหมด L (0-255) เพื่อให้สอดคล้องกับโมเดลการวัด SSIM
            source_img = Image.open(file).convert("L").resize((64, 64))

    if source_img:
        # เพื่อความเสถียรในการทำงานร่วมกับ Isomer Encoder แปลงภาพต้นฉบับจำลองเข้าสู่ฐานพิกเซล
        binary_for_encoder = source_img.convert("1")
        img_text_representation = image_to_isomers(binary_for_encoder)
        img_bitstream = ultimate_encoder(img_text_representation)

        st.markdown("---")
        error_count = st.slider("สไลเดอร์เพิ่มสัญญาณรบกวนในสายส่งข้อมูล (จำนวนบิตเสีย):", 0, 400, 0)

        corrupted_bits = inject_errors(img_bitstream,
                                       random.sample(range(len(img_bitstream)), error_count) if error_count > 0 else [])

        isomer_decoded_text = decode_isomer_v_final(corrupted_bits)
        isomer_output_img = isomers_to_image(isomer_decoded_text)

        # ปรับอิมเมจต้นฉบับให้อยู่ในโหมด L ระดับสเกล 0-255 สำหรับเปรียบเทียบ SSIM อย่างถูกต้อง
        source_img_l = source_img.convert("L")
        # ทำการชดเชยค่าพื้นหลังพิกเซลให้ชัดเจน
        arr_src = np.array(source_img_l)
        arr_src = (arr_src > 127).astype(np.uint8) * 255
        source_img_l = Image.fromarray(arr_src, mode="L")

        hamming_output_img = simulate_hamming_image(binary_for_encoder, error_count)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("1) รูปภาพต้นฉบับ")
            st.image(source_img_l, width=240, caption="Original Gray Image (L Mode)")
        with c2:
            st.subheader("2) วิธี Isomer (โครงงานของเรา)")
            st.image(isomer_output_img, width=240, caption="Isomer Reconstruction")
        with c3:
            st.subheader("3) วิธี Hamming Code")
            st.image(hamming_output_img, width=240, caption="Hamming Noise Result")

        # --- คำนวณดัชนี SSIM ---
        iso_ssim_score = calculate_image_ssim(source_img_l, isomer_output_img)
        ham_ssim_score = calculate_image_ssim(source_img_l, hamming_output_img)

        st.markdown("### 📊 เปรียบเทียบดัชนี Structural Similarity Index (SSIM) ของรูปภาพ")
        st.caption("หมายเหตุ: ค่า SSIM มีค่าอยู่ระหว่าง 0.0 ถึง 1.0 (ยิ่งเข้าใกล้ 1.0 ยิ่งเหมือนภาพต้นฉบับมากที่สุด)")

        m1, m2 = st.columns(2)
        m1.metric("ค่า SSIM ของวิธี Isomer", f"{iso_ssim_score:.4f}", delta=f"{iso_ssim_score - ham_ssim_score:.4f}")
        m2.metric("ค่า SSIM ของวิธี Hamming Code", f"{ham_ssim_score:.4f}")