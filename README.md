# Overview

## Articulating Goals

* We will create a highly modularized neural network framework capable of interchangeably swapping in various learning algorithms, activation functions, and structural configurations of the network.
* We will **NOT** use external neural network libraries like TensorFlow that might abstract away any processes, and we will instead minimally rely on libraries when it comes to the network's functionality.
* We would also like to visualize the network's testing process after it finishes training.
* We will land on a "best" configuration of the network parameters for each learning algorithm to conclude each section.

## Technologies Used

- **Python 3.11** – Primary programming language for neural network simulation and visualization
  - **numpy** – Matrix operations, vector math, and numerical utilities
  - **pyqtgraph** – Real-time GPU-accelerated visualization of neurons and activation flows
  - **PyQt5** – GUI framework for interactive training visualization and event control
  - **collections** – Used for accuracy tracking and performance aggregation by class
- **Git + GitHub** – Source control and remote project visibility
- **VSCode** – Primary development environment for all modules, debugging, and visualization
- **Paint.NET** – Used to create 64-bit .png files of letters which were procedurally converted to our dataset's letter matrices

# Notebook

## Step 1: Draft our Model

> We will train on three 8x8 binary encoded letters: A, C, and Z, with 28 training instances per letter type, simulating sparse-data high-dimension learning.

> Our network will have 64 neurons in its input layer.

> Our network will have 3 neurons in its output layer (each neuron representing the probability of the input vector being one of the three letters).

> This will be a standard neural network model (i.e., each neuron in layer A will connect to each neuron in layer B).

![images/NNDiagrams_Structure.png](images/NNDiagrams_Structure.png)

## Step 2: Create a Dataset

Let's use Paint.NET to create an 8x8 letter and save it as `letter_image.png`:

![images/pdnScreenshot.png](images/pdnScreenshot.png)

Next, let's write a script ([gen_data.py](data/gen_data.py)) to decode this letter, encode it as a binary Python matrix, and copy it to our clipboard:

```
# import libraries
from PIL import Image
import numpy as np
import pyperclip

# functions
def png_to_mat(path, threshold):
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
png_to_mat("letterImage.png", threshold=100)
```

After running this script, we paste the resulting matrix into our [letter_data.py](data/letter_data.py) script, where we can see 1s resembling the shape of the letter A:

![images/encodedLetter.png](images/encodedLetter.png)

We can do this until we have 30 hand-drawn letters per category saved in [letter_data.py](data/letter_data.py), with each letter matrix stored in one centralized list—the only list to be referenced by the rest of the model:

```
letterVariants = {
    "A": A_matrices,
    "C": C_matrices,
    "Z": Z_matrices,
}
```

## Step 3: Implement our Model


