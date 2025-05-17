# import libraries
from PyQt5 import QtWidgets
import pyqtgraph as pg
import numpy as np
from PyQt5.QtCore import QEventLoop

# initialize Qt application
app = QtWidgets.QApplication([])

# global widgets and state
win = pg.GraphicsLayoutWidget(title="Neural Network Visualization")
view = win.addViewBox()
view.setAspectLocked(True)

info_label = pg.TextItem(anchor=(0, 0), color=(255, 255, 255))  # top-left
info_label.setZValue(10)  # render above all nodes

continue_button = QtWidgets.QPushButton("Next")
event_loop = QEventLoop()

container = None
node_items = []
has_initialized = False


def initialize_network_canvas(layer_sizes):
    """
    Creates and displays the static neuron layout for the network.
    Only runs once.
    """
    global node_items, container, has_initialized

    if has_initialized:
        return
    has_initialized = True

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
    info_label.setPos(spacing_x + 10000, -spacing_y - 22000)
    view.addItem(info_label)

    continue_button.clicked.connect(event_loop.quit)

    layout = QtWidgets.QVBoxLayout()
    layout.addWidget(win)
    layout.addWidget(continue_button)

    container = QtWidgets.QWidget()
    container.setLayout(layout)
    container.show()


def update_activations(layer_outputs, actual_letter=None, predicted_letter=None, output_vector=None):
    """
    Updates node colors and status label based on current activations and prediction.
    """
    for layer_idx, activations in enumerate(layer_outputs):
        if layer_idx >= len(node_items):
            continue
        activations = np.array(activations).flatten()
        for neuron_idx, value in enumerate(activations):
            if neuron_idx >= len(node_items[layer_idx]):
                continue
            try:
                val_clipped = np.clip(value, 0, 1)
                intensity = int(val_clipped * 255)
                color = pg.mkColor(intensity, intensity, intensity)
                node_items[layer_idx][neuron_idx].setBrush(pg.mkBrush(color))
            except RuntimeError as e:
                print(f"Color assignment error at Layer {layer_idx}, Neuron {neuron_idx}: {e}")

    if actual_letter is not None and predicted_letter is not None and output_vector is not None:
        correct = actual_letter == predicted_letter
        result_color = "lime" if correct else "red"

        from data import letter_data
        letters = list(letter_data.letter_variants.keys())

        output_entries = [
            f"{letter}: {output_vector[i]:.2f}" for i, letter in enumerate(letters)
        ]
        output_str = "<br>".join(output_entries)

        html = f"""
            <div style="font-size:16pt; color:{result_color}">
                <b>Predicted:</b> '{predicted_letter}' | <b>Actual:</b> '{actual_letter}'
            </div>
            <div style="font-size:10pt; color:white; line-height:1.5">
                <b>Output Vector:</b><br>{output_str}
            </div>
        """
        try:
            info_label.setHtml(html)
        except RuntimeError as e:
            print(f"info_label assignment error: {e}")

    QtWidgets.QApplication.processEvents()


def wait_for_click():
    continue_button.setEnabled(True)
    continue_button.setText("Next")
    event_loop.exec_()
    continue_button.setEnabled(False)


def exec_app():
    app.exec_()
