# from starter_code.utils import load_segmentation
from utils import load_segmentation
import os

import numpy as np
import nibabel as nib

# this file path
dirname = os.path.dirname(__file__)
homeFolder = '/home/anya/Documents/kits19/new_tensorflow/'
datasetFolder = '/home/anya/Documents/kits19/data' # evaluating the test data
modelsFolder = os.path.join(homeFolder, 'models/202105060031')
predictionsFolder = os.path.join(modelsFolder, 'predictionstest')

outputFolder = 'results'
outPath = os.path.join(modelsFolder, outputFolder)

# create output directory
os.makedirs(outPath, exist_ok=True)

def evaluate(case_id, predictions):
    # Handle case of softmax output
    if len(predictions.shape) == 4:
        predictions = np.argmax(predictions, axis=-1)

    # Check predictions for type and dimensions
    if not isinstance(predictions, (np.ndarray, nib.Nifti1Image)):
        raise ValueError("Predictions must by a numpy array or Nifti1Image")
    if isinstance(predictions, nib.Nifti1Image):
        predictions = predictions.get_data()

    if not np.issubdtype(predictions.dtype, np.integer):
        predictions = np.round(predictions)
    predictions = predictions.astype(np.uint8)

    # Load ground truth segmentation
    gt = load_segmentation(case_id).get_data()

    # Make sure shape agrees with case
    if not predictions.shape == gt.shape:
        raise ValueError(
            ("Predictions for case {} have shape {} "
            "which do not match ground truth shape of {}").format(
                case_id, predictions.shape, gt.shape
            )
        )

    try:
        # Compute tumor+kidney Dice
        tk_pd = np.greater(predictions, 0)
        tk_gt = np.greater(gt, 0)
        tk_dice = 2*np.logical_and(tk_pd, tk_gt).sum()/(
            tk_pd.sum() + tk_gt.sum()
        )
    except ZeroDivisionError:
        return 0.0, 0.0

    try:
        # Compute tumor Dice
        tu_pd = np.greater(predictions, 1)
        tu_gt = np.greater(gt, 1)
        tu_dice = 2*np.logical_and(tu_pd, tu_gt).sum()/(
            tu_pd.sum() + tu_gt.sum()
        )
    except ZeroDivisionError:
        return tk_dice, 0.0

    return tk_dice #, tu_dice

# load cases
cases = os.listdir(datasetFolder)
cases.sort()
cases = cases[:]

for case in cases:
    # select data in to single array
    # pred = [] # predictions
    case_id = case
    P = load_segmentation(case).get_data()
    # predictionsPath = os.path.join(predictionsFolder, f'prediction_{case}.npz')
    # predictions = np.load(predictionsPath, None, True)
    # for slice in predictions.files:
    #     predarr = np.array(predictions[slice])
    #     predarr = predarr.reshape((1, 512, 512, 1))
    #     pred.append(predarr)
    #     pass
    
    # concatenate predictions
    # P = np.concatenate(pred)

    tk_dice = evaluate(case_id, P)
    print(f'tumor+kidney Dice for {case}: {tk_dice}')