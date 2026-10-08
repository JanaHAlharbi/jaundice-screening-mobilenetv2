# Newborn Jaundice Screening

A student project that classifies photos of newborns as **jaundice** or **normal**, using MobileNetV2 with transfer learning, and explains each prediction with a Grad-CAM heatmap.

> **This is a screening aid built for a course project. It is not a medical device and it does not give a diagnosis.** Jaundice must be confirmed with a bilirubin test.

Built during the Computer Vision Bootcamp by Saudi Digital Academy and atomcamp, at King Abdulaziz University.

## Why this project

Newborn jaundice is common. Most cases are mild, but a severe case that is missed can harm the baby. The usual check is a blood test. We wanted to see how far a photo-based screening model can go, and to understand where it fails.

## Dataset

We used **NJN (Normal and Jaundiced Newborns)**, a public dataset collected in one hospital.

- 760 colour images, all 1000 x 1000 pixels
- 755 images after removing 5 exact duplicates: 558 normal and 197 jaundice
- Labels are per image. There are no bounding boxes, masks or bilirubin values.

The images are **not included** in this repository. Download the dataset from its original source on Zenodo and place it in two folders named `normal` and `jaundice`.

## Approach

1. **Explore the data** before training: class counts, image sizes, colour comparison, sample images.
2. **Clean**: remove unreadable files and exact duplicates.
3. **Split**: 70% train, 15% validation, 15% test, stratified by class (528 / 113 / 114 images).
4. **Preprocess**: resize to 224 x 224 and normalize with ImageNet values.
5. **Augment** the training images: random crop, horizontal flip, small rotation and random erasing. We did not change colours, because colour is the sign of jaundice.
6. **Model**: MobileNetV2 pre-trained on ImageNet, with a new 2-output final layer.
7. **Train in two stages**: first the new layer only, then all layers with a smaller learning rate (fine-tuning), with early stopping.
8. **Handle class imbalance**: class weights and focal loss.
9. **Evaluate** on the held-out test set with recall, precision and the confusion matrix.
10. **Explain** predictions with Grad-CAM, and inspect the wrong predictions one by one.

## Results

Test set: 114 images (30 jaundice, 84 normal). Jaundice is the positive class.

| Experiment | What changed | Jaundice recall | Missed | False alarms | Accuracy |
|---|---|---|---|---|---|
| `baseline` | Full image, final layer only | 50% | 15 | 9 | 79% |
| `mask` | Skin isolated with colour thresholds | 63% | 11 | 13 | 79% |
| `mask_ft` | Mask, light fine-tuning | 63% | 11 | 19 | 74% |
| `mask_ft2` | Mask, full fine-tuning | 73% | 8 | 14 | 81% |
| `nomask_ft2_focal` | Full image, full fine-tuning, focal loss, random erasing | **83%** | **5** | 13 | **84%** |

For comparison, a model that always answers "normal" scores 74% accuracy and 0% recall.

| First model | Final model |
|---|---|
| ![Baseline confusion matrix](images/confusion_baseline.png) | ![Final confusion matrix](images/confusion_nomask_ft2_focal.png) |

## What we found

- **Accuracy was misleading.** The first model had 79% accuracy and missed half of the jaundice cases.
- **The model used shortcuts.** Grad-CAM showed the first model looking at the blanket. In the wrong predictions, yellow objects seemed to push the answer toward jaundice and white covers toward normal.
- **Whole-image colour does not separate the classes.** A simple yellowness measurement overlapped almost completely between the two classes.
- **Some jaundice photos were taken during phototherapy**, under blue light and with eye covers. Their colour is not representative, and the treatment equipment correlates with the label.
- **Teaching worked better than cleaning.** Keeping the full image and penalising mistakes (focal loss, random erasing, full fine-tuning) gave better recall than removing the background, with heatmaps on the baby's body.

## Limitations

- The test set is small: 30 jaundice images, so each image changes recall by more than 3%.
- Results varied between training runs, and experiments were compared on the same test set.
- The data comes from one hospital. Performance on other cameras, lighting and skin tones is unknown.
- Some babies have more than one photo, and the split is by image, not by baby.
- The last experiment changed several things at once, so we cannot yet say which change caused the gain.
- The model gives a class, not a bilirubin level.

## Repository contents

| File | Purpose |
|---|---|
| `jaundice_project.ipynb` | Full pipeline: data, training, evaluation, Grad-CAM, demo |
| `app.py` | Gradio demo that loads the trained weights |
| `requirements.txt` | Libraries needed by the demo |
| `images/` | Charts used in this README |

## How to run

**Training (Google Colab)**

1. Put the dataset in Google Drive at `MyDrive/NJN`, with the folders `normal` and `jaundice`.
2. Open `jaundice_project.ipynb` in Colab and select a T4 GPU runtime.
3. Run the cells from top to bottom. Results are saved to `MyDrive/jaundice_project`.

**Demo**

1. Place the trained weights file `best_nomask_ft2_focal.pt` next to `app.py`.
2. Install the libraries: `pip install -r requirements.txt gradio`
3. Run: `python app.py`

## Next steps

- Tune the decision threshold on the validation set to raise recall.
- Train a two-branch model that combines the full image with a baby-only view.
- Repeat experiments with several seeds and cross-validation.
- Test on a second dataset from another hospital.
- Export to ONNX and run the model on the device, so photos stay private.

## Team

- Name 1
- Name 2
- Name 3

## Acknowledgements

- Our instructor, [name], for the guidance that shaped the final approach.
- Saudi Digital Academy and atomcamp for the bootcamp.
- The authors of the NJN dataset for making it public.

## A note on AI assistance

We used an AI assistant (Claude) to help write code and explain concepts. We ran the experiments, inspected the wrong predictions, and made the decisions.
