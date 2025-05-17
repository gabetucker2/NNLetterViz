# Import libraries
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QEventLoop

# Initialize Qt application
app = QtWidgets.QApplication([])

# Create graphics window and viewbox
win = pg.GraphicsLayoutWidget(title="Neural Network Visualization")
view = win.addViewBox()
view.setAspectLocked(True)
# Global label for status text
info_label = pg.TextItem(anchor=(0, 0), color=(255, 255, 255))  # Anchor top-left
info_label.setZValue(10)  # Render above all nodes

# Storage for each layer's node graphics
node_items = []

def initialize_network_canvas(layer_sizes):
    """
    Prepares the canvas with a fixed set of neuron nodes based on network topology.
    """
    global node_items
    node_items = []
    view.clear()

    spacing_x = 5000
    spacing_y = 15000

    for layer_idx, layer_size in enumerate(layer_sizes):
        nodes = []
        for neuron_idx in range(layer_size):
            x = neuron_idx * spacing_x - (layer_size * spacing_x) / 2
            y = -layer_idx * spacing_y

            scatter = pg.ScatterPlotItem(
                [x], [y],
                size=15,
                brush=pg.mkBrush(128, 128, 128),
                pen=pg.mkPen(None)
            )

            view.addItem(scatter)
            nodes.append(scatter)
        node_items.append(nodes)

    view.setBackgroundColor((120, 150, 255))
    view.addItem(info_label)
    info_label.setPos(spacing_x + 10000, -spacing_y - 22000)

    win.show()

def update_activations(layer_outputs, actual_letter=None, predicted_letter=None, output_vector=None):
    for layer_idx, activations in enumerate(layer_outputs):
        activations = np.array(activations).flatten()
        for neuron_idx, value in enumerate(activations):
            val_clipped = np.clip(value, 0, 1)
            intensity = int(val_clipped * 255)
            color = pg.mkColor(intensity, intensity, intensity)
            node_items[layer_idx][neuron_idx].setBrush(pg.mkBrush(color))

    if actual_letter is not None and predicted_letter is not None and output_vector is not None:
        correct = actual_letter == predicted_letter
        color = "lime" if correct else "red"

        # Pull the letter labels dynamically from your dataset
        from data import letterData
        letters = list(letterData.letterVariants.keys())

        output_entries = [
            f"{letter}: {output_vector[i]:.2f}" for i, letter in enumerate(letters)
        ]
        output_str = "<br>".join(output_entries)

        html = f"""
            <div style="font-size:16pt; color:{color}">
                <b>Predicted:</b> '{predicted_letter}' | <b>Actual:</b> '{actual_letter}'
            </div>
            <div style="font-size:10pt; color:white; line-height:1.5">
                <b>Output Vector:</b><br>{output_str}
            </div>
        """
        info_label.setHtml(html)

    QtWidgets.QApplication.processEvents()

def exec_app():
    app.exec_()
