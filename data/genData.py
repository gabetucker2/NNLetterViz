# import libraries
from PIL import Image
import numpy as np
import pyperclip

# functions
def print_png_to_binary_matrix(path, threshold):
    img = Image.open(path).convert("L")
    img_array = np.array(img)
    binary_matrix = (img_array < threshold).astype(int)

    lines = []
    for row in binary_matrix:
        bitstring = ','.join(str(int(x)) for x in row)
        lines.append(f"        [{bitstring}],")
    result = "\n".join(lines)

    pyperclip.copy(result)
    print("Copied to clipboard")

# procedure
print_png_to_binary_matrix("letterImage.png", threshold=100)
