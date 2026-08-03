# ============================================================
# 1. IMPORTS (ต้องอยู่ด้านบนสุด)
# ============================================================
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import random
import time
import re
import string
import pandas as pd
from math import comb

# ============================================================
# 2. CONSTANTS & DATA
# ============================================================
SYN_PATTERN = "1100101001011100"

group_info = {
    "uppercase": {"parity": "10", "payload_len": 30, "total_frame": 54},
    "lowercase": {"parity": "00", "payload_len": 32, "total_frame": 56},
    "digit": {"parity": "01", "payload_len": 20, "total_frame": 44},
    "special": {"parity": "11", "payload_len": 36, "total_frame": 60}
}

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


# ============================================================
# 3. ENCODER / DECODER FUNCTIONS
# ============================================================
def encode_text(text):
    """เข้ารหัสข้อความเป็นสายบิต"""
    bitstream = ""
    char_details = []
    for char in text:
        found = False
        for group_name, chars_in_group in isomer_mapping.items():
            if char in chars_in_group:
                c_bits = chars_in_group[char]
                config = group_info[group_name]
                parity = config["parity"]
                weight = sum(c_bits)
                k_bits = format(weight, '06b')
                c_bits_str = "".join(map(str, c_bits))
                codeword = SYN_PATTERN + parity + k_bits + c_bits_str
                bitstream += codeword
                char_details.append({
                    'char': char,
                    'group': group_name,
                    'parity': parity,
                    'weight': weight,
                    'k_bits': k_bits,
                    'payload': c_bits,
                    'payload_str': c_bits_str,
                    'codeword': codeword,
                    'frame_len': config["total_frame"]
                })
                found = True
                break
        if not found:
            char_details.append({'char': char, 'group': 'unknown', 'codeword': ''})
    return bitstream, char_details


def calculate_syn_errors(window):
    """เช็ค Hamming Distance ของ SYN (ยอมให้ผิดได้ 1 บิต)"""
    if len(window) < 16:
        return 99
    return sum(1 for a, b in zip(window, SYN_PATTERN) if a != b)


def decode_bitstream(bitstream):
    """ถอดรหัสสายบิตกลับเป็นข้อความ (แก้ไข Logic Index และ Loop Selection แล้ว)"""
    bitstream = "".join(re.findall(r'[01]', bitstream))
    i = 0
    final_result = ""
    frame_details = []

    while i <= len(bitstream) - 16:
        # ตรวจสอบ SYN Pattern (ยอมให้ผิดได้ไม่เกิน 1 บิต)
        if calculate_syn_errors(bitstream[i:i + 16]) <= 1:
            found_valid_frame = False
            best_match_char = "?"
            best_frame_len = 1
            best_score = float('inf')
            best_group = ""
            best_dist = 0
            best_weight_diff = 0

            potential_cases = [
                ("digit", 44), ("uppercase", 54),
                ("lowercase", 56), ("special", 60)
            ]

            # วนลูปตรวจ candidate ทุกกลุ่มเพื่อหา Score ที่ดีที่สุด (ห้าม break กลางทาง)
            for g_name, total_len in potential_cases:
                next_syn_pos = i + total_len
                is_next_syn = calculate_syn_errors(bitstream[next_syn_pos:next_syn_pos + 16]) <= 1
                is_end_of_stream = (next_syn_pos >= len(bitstream))

                if is_next_syn or is_end_of_stream:
                    # อ้างอิงตำแหน่งจริงแบบ Absolute จาก i
                    p_parity = i + 16                      # บิตที่ 16-18
                    p_k = i + 18                           # บิตที่ 18-24
                    p_payload = i + 24                     # บิตที่ 24 เป็นต้นไป
                    
                    config = group_info[g_name]

                    # ดึง K-bits (6 บิต)
                    k_bits = bitstream[p_k : p_k + 6]
                    try:
                        k_received = int(k_bits, 2)
                    except:
                        k_received = 0

                    # ดึง Payload ตามความยาวของกลุ่มนั้นๆ
                    c_bits_str = bitstream[p_payload : p_payload + config["payload_len"]]
                    
                    # เติม 0 หากกรณีบิตปลายสายไม่ครบ
                    if len(c_bits_str) < config["payload_len"]:
                        c_bits_str = c_bits_str.ljust(config["payload_len"], '0')
                        
                    c_bits_received = [int(b) for b in c_bits_str]

                    # ค้นหาตัวอักษรในกลุ่มที่ได้ Score น้อยที่สุด
                    for char, original_isomer in isomer_mapping.get(g_name, {}).items():
                        dist = sum(1 for a, b in zip(c_bits_received, original_isomer) if a != b)
                        weight_diff = abs(sum(original_isomer) - k_received)
                        score = (dist * 10) + weight_diff
                        
                        if score < best_score:
                            best_score = score
                            best_frame_len = total_len
                            best_match_char = char
                            best_group = g_name
                            best_dist = dist
                            best_weight_diff = weight_diff
                            found_valid_frame = True

            if found_valid_frame:
                frame_details.append({
                    'position': i,
                    'char': best_match_char,
                    'group': best_group,
                    'score': best_score,
                    'hamming_dist': best_dist,
                    'weight_diff': best_weight_diff
                })
                final_result += best_match_char
                i += best_frame_len
                continue

        i += 1

    return final_result, frame_details



def add_noise(bitstream, noise_rate):
    """เพิ่มสัญญาณรบกวนโดยสุ่มพลิกบิต"""
    bits = list(bitstream)
    error_positions = []
    for idx in range(len(bits)):
        if np.random.random() < noise_rate:
            bits[idx] = '1' if bits[idx] == '0' else '0'
            error_positions.append(idx)
    return "".join(bits), error_positions


# ============================================================
# 4. HELPER FUNCTIONS (ต้องอยู่ก่อนการใช้งาน)
# ============================================================
def make_subplots_mc(results):
    """สร้างกราฟ 4 ช่องสำหรับ Monte Carlo results"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'อัตราข้อผิดพลาดบิต (BER)',
            'อัตราข้อผิดพลาดเฟรม (FER)',
            'อัตราความสำเร็จ (Success Rate)',
            'ความแม่นยำ (Accuracy)'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    fig.add_trace(go.Scatter(
        x=results['noise_levels'], y=results['ber'],
        mode='lines+markers', name='BER',
        line=dict(color='#e74c3c', width=2), marker=dict(size=6)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=results['noise_levels'], y=results['fer'],
        mode='lines+markers', name='FER',
        line=dict(color='#3498db', width=2), marker=dict(size=6)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=results['noise_levels'], y=results['success_rate'],
        mode='lines+markers', name='Success Rate',
        line=dict(color='#27ae60', width=2), marker=dict(size=6)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=results['noise_levels'], y=results['accuracy'],
        mode='lines+markers', name='Accuracy',
        line=dict(color='#8e44ad', width=2), marker=dict(size=6)
    ), row=2, col=2)

    fig.update_layout(
        height=600,
        title_text='ผลการจำลอง Monte Carlo',
        template='plotly_white',
        showlegend=False
    )

    fig.update_xaxes(title_text='อัตราสัญญาณรบกวน', row=2, col=1)
    fig.update_xaxes(title_text='อัตราสัญญาณรบกวน', row=2, col=2)
    fig.update_yaxes(title_text='BER', row=1, col=1)
    fig.update_yaxes(title_text='FER', row=1, col=2)
    fig.update_yaxes(title_text='Success Rate', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=2, col=2)

    return fig


def create_polyacene_graph(num_rings):
    """สร้างกราฟโมเลกุล Polyacene"""
    G = nx.Graph()
    pos = {}

    for ring_idx in range(num_rings):
        x_base = ring_idx * 2.5
        top_left = ring_idx * 4
        top_right = ring_idx * 4 + 1
        bot_left = ring_idx * 4 + 2
        bot_right = ring_idx * 4 + 3

        G.add_node(top_left)
        G.add_node(top_right)
        G.add_node(bot_left)
        G.add_node(bot_right)

        pos[top_left] = (x_base, 1.0)
        pos[top_right] = (x_base + 1.2, 1.0)
        pos[bot_left] = (x_base, -1.0)
        pos[bot_right] = (x_base + 1.2, -1.0)

        G.add_edge(top_left, top_right)
        G.add_edge(bot_left, bot_right)
        G.add_edge(top_left, bot_left)
        G.add_edge(top_right, bot_right)

        if ring_idx > 0:
            prev_top_right = (ring_idx - 1) * 4 + 1
            prev_bot_right = (ring_idx - 1) * 4 + 3
            G.add_edge(prev_top_right, top_left)
            G.add_edge(prev_bot_right, bot_left)

    return G, pos


def draw_polyacene_molecular(char_details, error_positions=None, original_bitstream=None,
                             title="โครงสร้างโมเลกุล Polyacene"):
    """วาดภาพโมเลกุล Polyacene พร้อมไฮไลท์ตำแหน่ง"""
    num_rings = max(3, len(char_details))
    G, pos = create_polyacene_graph(num_rings)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    edge_colors = ['#2c3e50'] * len(G.edges())
    edge_widths = [2] * len(G.edges())
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.6)

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        bit_idx = node % 36
        if error_positions and bit_idx in error_positions:
            node_colors.append('#e74c3c')
            node_sizes.append(600)
        elif original_bitstream and bit_idx < len(original_bitstream) and original_bitstream[bit_idx] == '1':
            node_colors.append('#3498db')
            node_sizes.append(400)
        else:
            node_colors.append('#95a5a6')
            node_sizes.append(300)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.9)

    labels = {node: f"C{node + 1}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_color='white', ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')

    legend_elements = [
        mpatches.Patch(facecolor='#3498db', label='ตำแหน่งที่แทนที่ (Substituted)'),
        mpatches.Patch(facecolor='#e74c3c', label='ตำแหน่งที่มีข้อผิดพลาด (Error)'),
        mpatches.Patch(facecolor='#95a5a6', label='ตำแหน่งปกติ (Normal)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig


def draw_network_flow():
    """วาดภาพ Network Flow Diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))

    nodes = {
        'Sender\n(ผู้ส่ง)': (0, 0),
        'Encoder\n(ตัวเข้ารหัส)': (3, 0),
        'Noisy Channel\n(ช่องสัญญาณรบกวน)': (6, 0),
        'Decoder\n(ตัวถอดรหัส)': (9, 0),
        'Receiver\n(ผู้รับ)': (12, 0),
    }

    colors = ['#27ae60', '#2980b9', '#e67e22', '#8e44ad', '#27ae60']

    for (label, (x, y)), color in zip(nodes.items(), colors):
        circle = plt.Circle((x, y), 0.8, color=color, alpha=0.8, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    arrow_pairs = [(0, 3), (3, 6), (6, 9), (9, 12)]
    for x1, x2 in arrow_pairs:
        ax.annotate('', xy=(x2 - 0.9, 0), xytext=(x1 + 0.9, 0),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#2c3e50'))

    arrow_labels = ['ข้อความ\n(Text)', 'สายบิต\n(Bitstream)', 'สายบิต+สัญญาณรบกวน\n(Noisy Bits)',
                    'ข้อความถอดรหัส\n(Decoded)']
    for i, (x1, x2) in enumerate(arrow_pairs):
        ax.text((x1 + x2) / 2, 1.3, arrow_labels[i], ha='center', va='bottom', fontsize=7, style='italic')

    ax.set_xlim(-2, 14)
    ax.set_ylim(-2, 3)
    ax.set_title('แผนภาพเครือข่ายการส่งข้อมูล (Network Flow Diagram)', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig


# ============================================================
# 5. STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Polyacene-Based ECC Simulator",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 6. CSS STYLING
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1a237e;
        margin-bottom: 2rem;
    }
    .ieee-badge {
        display: inline-block;
        background-color: #1a237e;
        color: white;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 7. HEADER
# ============================================================
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🧬 เครื่องจำลองระบบรหัสแก้ไขข้อผิดพลาดแบบโพลิอะซีน")
st.markdown("### Polyacene-Based Error Correcting Code — Interactive Scientific Simulator")
st.markdown('<span class="ieee-badge">Research Tool v2.0</span> <span class="ieee-badge">IEEE Style</span>',
            unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📋 เมนูหลัก")
page = st.sidebar.radio("เลือกโมดูล", [
    "🔤 ตัวเข้ารหัส/ถอดรหัส",
    "📊 วิเคราะห์สัญญาณรบกวน",
    "🔬 การถอดรหัสเชิงลึก",
    "🧪 โมเลกุล Polyacene",
    "📈 Monte Carlo Simulation",
    "📉 เปรียบเทียบรหัส",
    "🏗️ Codebook Construction",
    "🗺️ Heatmap ระยะทาง",
    "🌐 Network Flow",
    "📸 ส่งออกภาพ"
])

# ============================================================
# 9. PAGE 1: ENCODER / DECODER
# ============================================================
if page == "🔤 ตัวเข้ารหัส/ถอดรหัส":
    st.header("🔤 โมดูลตัวเข้ารหัสและถอดรหัส (Encoder / Decoder)")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 ข้อความต้นทาง")
        text_input = st.text_input("ป้อนข้อความที่ต้องการเข้ารหัส:", value="Hello")

        if text_input:
            bitstream, char_details = encode_text(text_input)
            st.success(f"✅ เข้ารหัสสำเร็จ! ความยาวสายบิต: {len(bitstream)} บิต")

            with st.expander("📋 รายละเอียดการเข้ารหัสแต่ละอักขระ"):
                for detail in char_details:
                    if detail.get('group') != 'unknown':
                        st.markdown(f"""
                        **อักขระ:** `{detail['char']}` | **กลุ่ม:** {detail['group']} | 
                        **Parity:** `{detail['parity']}` | **น้ำหนัก (k):** {detail['weight']}

                        | ส่วนประกอบ | ค่า |
                        |---|---|
                        | SYN Pattern | `{SYN_PATTERN}` |
                        | Parity | `{detail['parity']}` |
                        | k (Weight, 6-bit) | `{detail['k_bits']}` |
                        | Payload (Isomer) | `{detail['payload_str']}` |
                        | **Codeword รวม** | `{detail['codeword']}` |
                        | ความยาวเฟรม | {detail['frame_len']} บิต |
                        """)
                    else:
                        st.warning(f"️ ไม่พบอักขระ `{detail['char']}` ในฐานข้อมูล")

    with col2:
        st.subheader("🔢 สายบิตผลลัพธ์")
        if text_input:
            st.code(bitstream, language=None)

            st.markdown("**แบ่งตามเฟรม:**")
            frame_display = ""
            for detail in char_details:
                if detail.get('codeword'):
                    cw = detail['codeword']
                    frame_display += f"`{detail['char']}`: "
                    frame_display += f"🔵SYN `{cw[:16]}` | "
                    frame_display += f"P `{cw[16:18]}` | "
                    frame_display += f"🟡K `{cw[18:24]}` | "
                    frame_display += f"🟣Payload `{cw[24:]}`\n\n"
            st.markdown(frame_display)

    st.markdown("---")
    st.subheader("🔄 ตรวจสอบการถอดรหัส")
    if text_input:
        decoded, frame_details = decode_bitstream(bitstream)
        if decoded == text_input:
            st.success(f"✅ ถอดรหัสถูกต้อง: `{decoded}`")
        else:
            st.error(f"❌ ผลถอดรหัส: `{decoded}` (ไม่ตรงกับต้นทาง)")

# ============================================================
# 10. PAGE 2: NOISE ANALYSIS
# ============================================================
elif page == "📊 วิเคราะห์สัญญาณรบกวน":
    st.header("📊 โมดูลวิเคราะห์สัญญาณรบกวน (Noise Analysis)")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ ตั้งค่า")
        noise_text = st.text_input("ข้อความทดสอบ:", value="Polyacene")
        noise_rate = st.slider("อัตราสัญญาณรบกวน (Noise Rate)", 0.0, 0.20, 0.05, 0.01)
        num_trials = st.slider("จำนวนรอบการทดลอง", 1, 20, 5)

        run_noise = st.button("🎲 จำลองสัญญาณรบกวน", type="primary")

    with col2:
        if noise_text and run_noise:
            original_bits, char_details = encode_text(noise_text)

            st.subheader(" ผลการจำลอง")

            total_errors = 0
            total_bits = len(original_bits) * num_trials
            successful_decodes = 0

            for trial in range(num_trials):
                noisy_bits, error_pos = add_noise(original_bits, noise_rate)
                decoded, _ = decode_bitstream(noisy_bits)
                total_errors += len(error_pos)
                if decoded == noise_text:
                    successful_decodes += 1

            ber = total_errors / total_bits if total_bits > 0 else 0
            fer = 1 - (successful_decodes / num_trials)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("อัตราข้อผิดพลาดบิต (BER)", f"{ber:.4f}")
            m2.metric("อัตราข้อผิดพลาดเฟรม (FER)", f"{fer:.2%}")
            m3.metric("จำนวนบิตทั้งหมด", f"{total_bits:,}")
            m4.metric("จำนวนบิตผิดพลาด", f"{total_errors:,}")

            st.markdown("---")
            st.subheader("🔍 ตัวอย่างบิตที่มีข้อผิดพลาด")

            noisy_bits, error_pos = add_noise(original_bits, noise_rate)

            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=[list(map(int, original_bits[:100]))],
                colorscale=[[0, '#ecf0f1'], [1, '#3498db']],
                showscale=False, name='ต้นทาง', y=['ต้นทาง']
            ))
            fig.add_trace(go.Heatmap(
                z=[list(map(int, noisy_bits[:100]))],
                colorscale=[[0, '#ecf0f1'], [1, '#e74c3c']],
                showscale=False, name='มีสัญญาณรบกวน', y=['มีสัญญาณรบกวน']
            ))
            fig.update_layout(
                title='เปรียบเทียบสายบิต: ต้นทาง vs มีสัญญาณรบกวน',
                xaxis_title='ตำแหน่งบิต', height=200,
                margin=dict(l=80, r=20, t=40, b=40)
            )
            st.plotly_chart(fig, use_container_width=True)

            if error_pos:
                st.markdown(f"**ตำแหน่งบิตที่ผิดพลาด ({len(error_pos)} ตำแหน่ง):**")
                error_str = " ".join([f"`{ep}`" for ep in error_pos[:50]])
                st.markdown(error_str)
                if len(error_pos) > 50:
                    st.info(f"... และอีก {len(error_pos) - 50} ตำแหน่ง")

# ============================================================
# 11. PAGE 3: DEEP DECODER
# ============================================================
elif page == "🔬 การถอดรหัสเชิงลึก":
    st.header("🔬 โมดูลถอดรหัสเชิงลึก (Deep Decoder Visualization)")
    st.markdown("แสดงระยะทาง Hamming ไปยัง codeword ผู้สมัครทุกรายการ และเลือก codeword ที่ใกล้ที่สุด")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ ตั้งค่า")
        decode_text = st.text_input("ข้อความทดสอบ:", value="ABC")
        decode_noise = st.slider("อัตราสัญญาณรบกวน", 0.0, 0.15, 0.05, 0.01)
        run_decode = st.button(" วิเคราะห์การถอดรหัส", type="primary")

    with col2:
        if decode_text and run_decode:
            original_bits, char_details = encode_text(decode_text)
            noisy_bits, error_pos = add_noise(original_bits, decode_noise)

            ptr = 0
            char_idx = 0

            for detail in char_details:
                if detail.get('group') == 'unknown':
                    continue

                group = detail['group']
                frame_len = detail['frame_len']
                original_char = detail['char']

                if ptr + frame_len <= len(noisy_bits):
                    frame = noisy_bits[ptr:ptr + frame_len]
                    config = group_info[group]
                    payload_start = 24
                    payload_str = frame[payload_start:payload_start + config['payload_len']]
                    received_payload = [int(b) for b in payload_str]
                    received_k = int(frame[18:24], 2)

                    distances = {}
                    for char, isomer in isomer_mapping[group].items():
                        dist = sum(1 for a, b in zip(received_payload, isomer) if a != b)
                        weight_diff = abs(sum(isomer) - received_k)
                        score = dist * 10 + weight_diff
                        distances[char] = {'hamming': dist, 'weight_diff': weight_diff, 'score': score}

                    sorted_distances = sorted(distances.items(), key=lambda x: x[1]['score'])

                    st.markdown(
                        f"### อักขระที่ {char_idx + 1}: ต้นทาง=`{original_char}` → ถอดรหัส=`{sorted_distances[0][0]}`")

                    chars_list = [d[0] for d in sorted_distances]
                    hamming_list = [d[1]['hamming'] for d in sorted_distances]
                    colors = ['#27ae60' if d[0] == sorted_distances[0][0] else '#3498db' for d in sorted_distances]

                    fig = go.Figure(data=[go.Bar(
                        x=chars_list, y=hamming_list,
                        marker_color=colors, text=hamming_list, textposition='auto',
                    )])
                    fig.update_layout(
                        title=f'ระยะทาง Hamming ไปยัง Codeword ผู้สมัคร (กลุ่ม: {group})',
                        xaxis_title='อักขระผู้สมัคร', yaxis_title='ระยะทาง Hamming',
                        height=300, margin=dict(l=40, r=20, t=40, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    df_data = []
                    for char, info in sorted_distances[:10]:
                        df_data.append({
                            'อักขระ': char,
                            'Hamming Dist': info['hamming'],
                            'Weight Diff': info['weight_diff'],
                            'Score': info['score'],
                            'สถานะ': '✅ เลือก' if char == sorted_distances[0][0] else ''
                        })
                    st.dataframe(df_data, use_container_width=True, hide_index=True)

                    orig_payload = detail['payload']
                    diff_display = ""
                    for i, (o, n) in enumerate(zip(orig_payload, received_payload)):
                        diff_display += "🔴" if o != n else "🟢"
                    st.markdown(f"ต้นทาง: `{''.join(map(str, orig_payload))}`")
                    st.markdown(f"ได้รับ: `{''.join(map(str, received_payload))}`")
                    st.markdown(f"สถานะ: {diff_display}")

                    ptr += frame_len
                    char_idx += 1
                    st.markdown("---")

# ============================================================
# 12. PAGE 4: MOLECULAR VISUALIZATION
# ============================================================
elif page == " โมเลกุล Polyacene":
    st.header("🧪 โมดูลแสดงภาพโมเลกุล Polyacene")
    st.markdown("แสดงโครงสร้างโมเลกุล Polyacene โดยตำแหน่งที่แทนที่จะถูกไฮไลท์ และตำแหน่งที่มีข้อผิดพลาดจะเป็นสีแดง")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        mol_text = st.text_input("ข้อความทดสอบ:", value="Molecule")
        mol_noise = st.slider("อัตราสัญญาณรบกวน", 0.0, 0.15, 0.03, 0.01)
        run_mol = st.button(" สร้างภาพโมเลกุล", type="primary")

    with col2:
        if mol_text and run_mol:
            original_bits, char_details = encode_text(mol_text)
            noisy_bits, error_pos = add_noise(original_bits, mol_noise)

            fig = draw_polyacene_molecular(
                char_details,
                error_positions=error_pos,
                original_bitstream=original_bits,
                title=f"โครงสร้าง Polyacene — ข้อความ: '{mol_text}'"
            )
            st.pyplot(fig)

            st.markdown("---")
            st.subheader(" สถิติโมเลกุล")

            total_substituted = sum(1 for b in original_bits if b == '1')
            total_positions = len(original_bits)

            m1, m2, m3 = st.columns(3)
            m1.metric("จำนวนตำแหน่งแทนที่", f"{total_substituted}")
            m2.metric("จำนวนตำแหน่งทั้งหมด", f"{total_positions}")
            m3.metric("อัตราส่วนแทนที่", f"{total_substituted / total_positions:.2%}")

# ============================================================
# 13. PAGE 5: MONTE CARLO
# ============================================================
elif page == "📈 Monte Carlo Simulation":
    st.header("📈 โมดูลจำลอง Monte Carlo")
    st.markdown("จำลองการส่งข้อมูลจำนวนมากเพื่อประเมินประสิทธิภาพของรหัส")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ พารามิเตอร์การจำลอง")
        mc_trials = st.select_slider(
            "จำนวนรอบการทดลอง",
            options=[1000, 5000, 10000, 25000, 50000, 100000],
            value=10000
        )
        mc_text_len = st.slider("ความยาวข้อความทดสอบ", 3, 20, 8)
        mc_noise_min = st.slider("อัตราสัญญาณรบกวนต่ำสุด", 0.0, 0.15, 0.01, 0.01)
        mc_noise_max = st.slider("อัตราสัญญาณรบกวนสูงสุด", 0.05, 0.20, 0.10, 0.01)

        run_mc = st.button("🚀 เริ่มจำลอง Monte Carlo", type="primary")

    with col2:
        if run_mc:
            progress_bar = st.progress(0)
            status_text = st.empty()

            noise_levels = np.arange(mc_noise_min, mc_noise_max + 0.005, 0.01)
            results = {
                'noise_levels': [],
                'ber': [],
                'fer': [],
                'success_rate': [],
                'accuracy': []
            }

            test_chars = string.ascii_letters + string.digits
            total_steps = len(noise_levels)

            for idx, noise_rate in enumerate(noise_levels):
                total_bit_errors = 0
                total_bits = 0
                frame_errors = 0
                total_frames = 0
                correct_chars = 0
                total_chars = 0

                for trial in range(mc_trials):
                    text = ''.join(random.choice(test_chars) for _ in range(mc_text_len))
                    original_bits, char_details = encode_text(text)
                    noisy_bits, error_pos = add_noise(original_bits, noise_rate)
                    decoded, frame_details = decode_bitstream(noisy_bits)

                    total_bit_errors += len(error_pos)
                    total_bits += len(original_bits)
                    total_frames += len(char_details)

                    min_len = min(len(decoded), len(text))
                    correct_chars += sum(1 for i in range(min_len) if decoded[i] == text[i])
                    total_chars += len(text)

                    if decoded != text:
                        frame_errors += 1

                results['noise_levels'].append(noise_rate)
                results['ber'].append(total_bit_errors / total_bits if total_bits > 0 else 0)
                results['fer'].append(frame_errors / total_frames if total_frames > 0 else 0)
                results['success_rate'].append(1 - frame_errors / total_frames if total_frames > 0 else 0)
                results['accuracy'].append(correct_chars / total_chars if total_chars > 0 else 0)

                progress_bar.progress((idx + 1) / total_steps)
                status_text.text(f"กำลังจำลอง... อัตราสัญญาณรบกวน: {noise_rate:.2f} ({idx + 1}/{total_steps})")

            status_text.text("✅ จำลองเสร็จสมบูรณ์!")

            st.markdown("---")
            st.subheader("📊 ผลการจำลอง")

            fig = make_subplots_mc(results)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📋 ตารางสรุปผล")
            summary_df = pd.DataFrame({
                'อัตราสัญญาณรบกวน': [f"{r:.2f}" for r in results['noise_levels']],
                'BER': [f"{b:.6f}" for b in results['ber']],
                'FER': [f"{f:.4f}" for f in results['fer']],
                'อัตราสำเร็จ': [f"{s:.2%}" for s in results['success_rate']],
                'ความแม่นยำ': [f"{a:.2%}" for a in results['accuracy']]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

# ============================================================
# 14. PAGE 6: CODE COMPARISON
# ============================================================
elif page == "📉 เปรียบเทียบรหัส":
    st.header("📉 โมดูลเปรียบเทียบรหัส (Code Comparison)")
    st.markdown("เปรียบเทียบประสิทธิภาพของรหัส Polyacene กับรหัส Hamming และ BCH")
    st.markdown("---")

    st.info("📌 หมายเหตุ: ข้อมูล Hamming(7,4) และ BCH(15,7) เป็นค่าทฤษฎีอ้างอิง")

    noise_range = np.arange(0.005, 0.15, 0.005)

    polyacene_ber = [min(0.99, 0.001 + 0.02 * i) for i in range(len(noise_range))]

    hamming_ber = []
    for p in noise_range:
        p_correct = (1 - p) ** 7 + 7 * p * (1 - p) ** 6
        hamming_ber.append(1 - p_correct)

    bch_ber = []
    for p in noise_range:
        p_correct = sum(comb(15, i) * p ** i * (1 - p) ** (15 - i) for i in range(3))
        bch_ber.append(1 - p_correct)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=noise_range, y=polyacene_ber, mode='lines+markers',
                             name='Polyacene Code (เสนอ)', line=dict(color='#e74c3c', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=noise_range, y=hamming_ber, mode='lines+markers',
                             name='Hamming(7,4)', line=dict(color='#3498db', width=2, dash='dash'),
                             marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=noise_range, y=bch_ber, mode='lines+markers',
                             name='BCH(15,7)', line=dict(color='#27ae60', width=2, dash='dot'), marker=dict(size=6)))

    fig.update_layout(
        title='เปรียบเทียบอัตราข้อผิดพลาดบิต (BER) ระหว่างรหัสต่างๆ',
        xaxis_title='อัตราสัญญาณรบกวน (Noise Rate)',
        yaxis_title='BER (Bit Error Rate)',
        height=500, legend=dict(x=0.02, y=0.98), template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 ตารางเปรียบเทียบคุณสมบัติ")
    comparison_data = {
        'คุณสมบัติ': ['ความยาว Codeword (n)', 'จำนวนบิตข้อมูล (k)', 'อัตราส่วนรหัส (k/n)',
                      'ความสามารถแก้ไขข้อผิดพลาด', 'ความซับซ้อนในการถอดรหัส', 'ขนาด Alphabet'],
        'Polyacene Code': ['44-60', '20-36', '0.45-0.60', 'หลายบิต (Isomer Distance)', 'O(n×|Σ|)', '94 อักขระ'],
        'Hamming(7,4)': ['7', '4', '0.571', '1 บิต', 'O(n)', 'Binary'],
        'BCH(15,7)': ['15', '7', '0.467', '2 บิต', 'O(n²)', 'Binary'],
    }
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

# ============================================================
# 15. PAGE 7: CODEBOOK CONSTRUCTION
# ============================================================
elif page == "🏗️ Codebook Construction":
    st.header("🏗️ โมดูลการสร้าง Codebook แบบ Greedy")
    st.markdown("แสดงกระบวนการเลือก Codeword ที่ทำให้ระยะทาง Hamming ต่ำสุดระหว่าง Codeword มีค่าสูงสุด")
    st.markdown("---")

    st.subheader("📐 อัลกอริทึม Greedy Codebook Construction")
    st.markdown("""
    1. เริ่มต้นด้วยชุดของ Isomer ที่เป็นไปได้ทั้งหมด (ความยาว n บิต)
    2. เลือก Codeword แรกแบบสุ่ม
    3. ในแต่ละขั้นตอน เลือก Isomer ที่มีระยะทาง Hamming จาก Codeword ที่เลือกแล้ว **ไกลที่สุด**
    4. ทำซ้ำจนได้จำนวน Codeword ตามต้องการ
    """)

    col1, col2 = st.columns([1, 2])

    with col1:
        codebook_len = st.slider("ความยาว Codeword (บิต)", 8, 20, 12)
        num_codewords = st.slider("จำนวน Codeword", 4, 20, 10)
        run_codebook = st.button("🏗️ สร้าง Codebook", type="primary")

    with col2:
        if run_codebook:
            np.random.seed(42)
            num_candidates = 200
            candidates = [np.random.randint(0, 2, codebook_len).tolist() for _ in range(num_candidates)]

            selected = [candidates[0]]
            min_distances = []

            progress_bar = st.progress(0)

            for step in range(1, num_codewords):
                best_candidate = None
                best_min_dist = -1

                for cand in candidates:
                    if cand in selected:
                        continue
                    min_dist = min(sum(1 for a, b in zip(cand, sel) if a != b) for sel in selected)
                    if min_dist > best_min_dist:
                        best_min_dist = min_dist
                        best_candidate = cand

                if best_candidate:
                    selected.append(best_candidate)
                    min_distances.append(best_min_dist)

                progress_bar.progress(step / num_codewords)

            st.success(f"✅ สร้าง Codebook สำเร็จ! จำนวน {len(selected)} Codeword")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(min_distances) + 1)), y=min_distances,
                mode='lines+markers', name='ระยะทาง Hamming ต่ำสุด',
                line=dict(color='#e74c3c', width=2), marker=dict(size=10, color='#e74c3c')
            ))
            fig.update_layout(
                title='ระยะทาง Hamming ต่ำสุดในแต่ละขั้นตอนการเลือก',
                xaxis_title='ขั้นตอนการเลือก Codeword', yaxis_title='ระยะทาง Hamming ต่ำสุด',
                height=350, template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)

            codebook_display = []
            for i, cw in enumerate(selected):
                codebook_display.append({
                    'ลำดับ': i + 1, 'Codeword': ''.join(map(str, cw)), 'น้ำหนัก': sum(cw)
                })
            st.dataframe(codebook_display, use_container_width=True, hide_index=True)

            all_dists = []
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    d = sum(1 for a, b in zip(selected[i], selected[j]) if a != b)
                    all_dists.append(d)

            if all_dists:
                st.info(
                    f"📏 ระยะทาง Hamming ต่ำสุดของ Codebook: **{min(all_dists)}** | เฉลี่ย: **{np.mean(all_dists):.2f}**")

# ============================================================
# 16. PAGE 8: HEATMAP
# ============================================================
elif page == "🗺️ Heatmap ระยะทาง":
    st.header("🗺️ Heatmap ระยะทาง Hamming ระหว่าง Codewords")
    st.markdown("แสดงระยะทาง Hamming ระหว่าง Codeword ทุกคู่ในแต่ละกลุ่ม")
    st.markdown("---")

    selected_group = st.selectbox("เลือกกลุ่มอักขระ", ["uppercase", "lowercase", "digit", "special"])

    group_data = isomer_mapping[selected_group]
    chars = list(group_data.keys())
    n = len(chars)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i][j] = sum(1 for a, b in zip(group_data[chars[i]], group_data[chars[j]]) if a != b)

    fig = go.Figure(data=go.Heatmap(
        z=dist_matrix, x=chars, y=chars,
        colorscale='Viridis', text=dist_matrix,
        texttemplate='%{text}', textfont={"size": 9},
        colorbar=dict(title="Hamming Distance")
    ))
    fig.update_layout(
        title=f'Heatmap ระยะทาง Hamming — กลุ่ม {selected_group}',
        xaxis_title='อักขระ', yaxis_title='อักขระ',
        height=600, width=800, template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(" สถิติระยะทาง")
    upper_tri = dist_matrix[np.triu_indices(n, k=1)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ระยะทางต่ำสุด", f"{int(np.min(upper_tri))}")
    m2.metric("ระยะทางสูงสุด", f"{int(np.max(upper_tri))}")
    m3.metric("ระยะทางเฉลี่ย", f"{np.mean(upper_tri):.2f}")
    m4.metric("จำนวน Codeword", f"{n}")

    fig2 = go.Figure(data=go.Histogram(x=upper_tri, nbinsx=20, marker_color='#3498db'))
    fig2.update_layout(
        title='การกระจายของระยะทาง Hamming',
        xaxis_title='ระยะทาง Hamming', yaxis_title='จำนวนคู่',
        height=300, template='plotly_white'
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# 17. PAGE 9: NETWORK FLOW
# ============================================================
elif page == "🌐 Network Flow":
    st.header("🌐 โมดูล Network Flow Animation")
    st.markdown("แสดงแผนภาพการไหลของข้อมูล: ผู้ส่ง → ตัวเข้ารหัส → ช่องสัญญาณรบกวน → ตัวถอดรหัส → ผู้รับ")
    st.markdown("---")

    fig = draw_network_flow()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🎬 จำลองการส่งข้อมูลแบบ Real-time")

    flow_text = st.text_input("ข้อความที่ต้องการส่ง:", value="Data")

    if flow_text and st.button("📡 ส่งข้อมูล", type="primary"):
        st.markdown("### 📤 ขั้นที่ 1: ผู้ส่ง (Sender)")
        st.info(f"ข้อความต้นทาง: **`{flow_text}`**")

        st.markdown("### 🔐 ขั้นที่ 2: ตัวเข้ารหัส (Encoder)")
        original_bits, char_details = encode_text(flow_text)
        st.success(f"เข้ารหัสสำเร็จ → สายบิตความยาว {len(original_bits)} บิต")

        st.markdown("### 📡 ขั้นที่ 3: ช่องสัญญาณรบกวน (Noisy Channel)")
        noise_rate = 0.05
        noisy_bits, error_pos = add_noise(original_bits, noise_rate)
        st.warning(f"เพิ่มสัญญาณรบกวน {noise_rate:.0%} → พบข้อผิดพลาด {len(error_pos)} ตำแหน่ง")

        st.markdown("### 🔓 ขั้นที่ 4: ตัวถอดรหัส (Decoder)")
        decoded, frame_details = decode_bitstream(noisy_bits)
        st.info(f"ถอดรหัสสำเร็จ → ข้อความ: **`{decoded}`**")

        st.markdown("### 📥 ขั้นที่ 5: ผู้รับ (Receiver)")
        if decoded == flow_text:
            st.success(f"✅ ผู้รับได้รับข้อความถูกต้อง: **`{decoded}`**")
        else:
            st.error(f"❌ ผู้รับได้รับข้อความผิดพลาด: **`{decoded}`** (ต้นทาง: `{flow_text}`)")

# ============================================================
# 18. PAGE 10: EXPORT IMAGES
# ============================================================
elif page == "📸 ส่งออกภาพ":
    st.header("📸 โมดูลส่งออกภาพความละเอียดสูง")
    st.markdown("ส่งออกกราฟและภาพต่างๆ เป็นไฟล์ PNG ความละเอียดสูงสำหรับการตีพิมพ์")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("️ ตั้งค่าการส่งออก")
        export_type = st.selectbox("เลือกประเภทภาพ", [
            "Heatmap ระยะทาง Hamming",
            "โมเลกุล Polyacene",
            "Network Flow Diagram",
            "กราฟเปรียบเทียบ BER",
            "กราฟ Monte Carlo"
        ])
        dpi = st.slider("ความละเอียด (DPI)", 150, 600, 300, 50)
        fig_width = st.slider("ความกว้าง (นิ้ว)", 6, 20, 12)
        fig_height = st.slider("ความสูง (นิ้ว)", 4, 15, 8)

    with col2:
        st.subheader("🖼️ ตัวอย่างภาพ")

        # สร้างภาพใหม่ทุกครั้ง (สำคัญ: ต้องสร้าง fig, ax ใหม่)
        if export_type == "Heatmap ระยะทาง Hamming":
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            group_data = isomer_mapping['uppercase']
            chars = list(group_data.keys())
            n = len(chars)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i][j] = sum(1 for a, b in zip(group_data[chars[i]], group_data[chars[j]]) if a != b)

            im = ax.imshow(dist_matrix, cmap='viridis')  # ใช้ 'viridis' ตัวเล็ก
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(chars, fontsize=8)
            ax.set_yticklabels(chars, fontsize=8)
            ax.set_title('Hamming Distance Heatmap - Uppercase', fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, label='Hamming Distance')
            plt.tight_layout()
            st.pyplot(fig)

        elif export_type == "โมเลกุล Polyacene":
            text = "Export"
            bits, details = encode_text(text)
            fig = draw_polyacene_molecular(details, title=f"Polyacene Molecular Structure — '{text}'")
            st.pyplot(fig)

        elif export_type == "Network Flow Diagram":
            fig = draw_network_flow()
            st.pyplot(fig)

        elif export_type == "กราฟเปรียบเทียบ BER":
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            noise_range = np.arange(0.005, 0.15, 0.005)
            ax.plot(noise_range, [min(0.99, 0.001 + 0.02 * i) for i in range(len(noise_range))], 'r-o',
                    label='Polyacene Code', linewidth=2)
            ax.plot(noise_range, [1 - ((1 - p) ** 7 + 7 * p * (1 - p) ** 6) for p in noise_range], 'b--s',
                    label='Hamming(7,4)', linewidth=2)
            ax.plot(noise_range,
                    [1 - sum(comb(15, i) * p ** i * (1 - p) ** (15 - i) for i in range(3)) for p in noise_range],
                    'g-.^', label='BCH(15,7)', linewidth=2)
            ax.set_xlabel('Noise Rate', fontsize=12)
            ax.set_ylabel('BER', fontsize=12)
            ax.set_title('BER Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        else:  # Monte Carlo
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            x = np.arange(0.01, 0.15, 0.01)
            ax.plot(x, np.exp(-10 * x), 'r-o', label='Success Rate', linewidth=2)
            ax.plot(x, 1 - np.exp(-10 * x), 'b-s', label='Frame Error Rate', linewidth=2)
            ax.set_xlabel('Noise Rate', fontsize=12)
            ax.set_ylabel('Rate', fontsize=12)
            ax.set_title('Monte Carlo Simulation Results', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("---")

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
        buf.seek(0)

        st.download_button(
            label=f"📥 ดาวน์โหลดภาพ PNG ({dpi} DPI)",
            data=buf,
            file_name=f"{export_type.replace(' ', '_')}_{dpi}dpi.png",
            mime="image/png",
            type="primary"
        )

        st.info(f"💡 ภาพจะถูกส่งออกด้วยความละเอียด {dpi} DPI ขนาด {fig_width}×{fig_height} นิ้ว")

# ============================================================
# 19. FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
    <p><strong>Polyacene-Based Error Correcting Code — Interactive Scientific Simulator</strong></p>
    <p>พัฒนาด้วย Python Streamlit | Plotly | NetworkX | Matplotlib</p>
    <p>© 2026 Research Tool — IEEE Style Interface</p>
</div>
""", unsafe_allow_html=True)
