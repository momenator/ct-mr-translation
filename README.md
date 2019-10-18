# CT to MR Spinal Translation

What is in this repo? Preprocessing pipeline to process VISCERAL scans for the Spinal CT to MR Translation project and evaluation.

### Quickstart

1. Download the visceral dataset and put 36 CT and MRI scans (72 in total) in folder data/visceral/train. For the test set,
put the remaining scans in folder data/visceral/test.

2. Run ```python prep_data.py``` to prepare the training and test sets.

3. There should be trainA and trainB folders in data/visceral containing numpy arrays. Each numpy array contains a patch from CT
or MRI slice.

4. Transfer the data to the CycleGAN repository and start the training there. The scripts move_data_visceral.sh might be useful for this task.

### Evaluation

Some results from this project may be viewed in results.ipynb.

