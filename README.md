# Multi-domain-image-dataset-diversity-and-complexity-measures-for-multi-object-detection-program

## Model Setup

Before running the project, create a folder named `model` at the same level as the project files and download the following two model files into this folder:

* [RT_GCDIoU.pt](https://drive.google.com/file/d/14OodAin2pcMCwjjhiGGJ_r1rBpOnzUvY/view)
* [YOLO_GDCIoU.pt](https://drive.google.com/file/d/15sSr7fyBSDJVw3tLZdOCABSuSy4x2S2E/view)

The folder structure should look like this:

```text
project/
├── model/
│   ├── RT_GCDIoU.pt
│   └── YOLO_GDCIoU.pt
├── ...
```
Make sure both model files are placed inside the `model` folder before running the project.

## User Interface

The graphical user interface is designed using **Qt Designer** and is stored in the `form.ui` file.

If you modify the user interface using Qt Designer, you must regenerate the Python UI file before running the project.

After making changes to `form.ui`, run the following command from the project directory:

```bash
python -m PyQt5.uic.pyuic form.ui -o New_window.py
```

This command converts the `form.ui` file into the Python file `New_window.py`.

**Important:** Whenever `form.ui` is modified, run the command above again to update `New_window.py`.

## Running the Project

The main Python file of the project is `GasDis.py`.

After completing the model setup and generating `New_window.py`, run:

```bash
python Dataset_diversity.py
```

Make sure Python, PyQt5, and all other required dependencies have been installed before running the project.
