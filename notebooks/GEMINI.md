# Training Pipeline Context 

We should proceed on structuring our notebooks with the latest (v4) format. And here are the context for writing the notebook.

## Google Drive
* The root directory of the whole training and optimization pipeline in Google Colab environment will be at `/content/drive/MyDrive/PawikanSentinel`, considering that the Google Drive was mounted. 
* The root will have subfolders: `datasets`, `notebooks`, `models`, `logs`.
* And from the `datasets` directory will contain the `pawikan_dataset.zip`, a combined labelled dataset from the GTST-2023 and SeaTurtleID2022 datasets.

## Notebok Structure
- [ ] **Cell 1: Setup.** Mount the Google Drive, `cd` into the `yolov5` directory, and `pip install` any dependencies. 
- [ ] **Cell 2: Data Preparation.** Copy your `pawikan_dataset.zip` from Drive to the Colab instance and unzip it.
- [ ] **Cell 3: Training.** Run the `yolov5/train.py` command. Crucially, point the `--weights` argument to your last checkpoint saved on Google Drive to resume training.
- [ ] **Cell 4: Model Optimization.** Once training is complete, run our `optimization_pipeline.py` script. It will find the `best.pt` from the latest training run, convert it to ONNX, and then to TFLite.
- [ ] **Cell 5: Save Artifacts.** Copy the final `.tflite` model, the `.onnx` model, and any training logs from the Colab `runs` directory back to the `models` and `logs` folders in Google Drive.
