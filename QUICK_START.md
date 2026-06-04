# Quick Start Guide - Real X-Ray Image Validation

## Your Dataset

✓ **199 X-ray images** are ready in `training_images/`
- Sourced from 5 trays (Lot 2: 260401469)
- Standard PNG format, duplicates removed

## Step 1: Prepare Training Data

Generate annotation template from your images:

```bash
python prepare_training_data.py
```

**Output:**
- Creates `training_manifest_template.json`
- Shows image statistics
- Lists images ready for labeling

## Step 2: Label Your Images

Edit `training_manifest_template.json` and mark each image:

```json
{
  "samples": [
    {
      "image_path": "training_images/101417395279.png",
      "label": "valid",  // ← Change this
      "metadata": {...}
    },
    {
      "image_path": "training_images/210243928143.png", 
      "label": "invalid",  // ← Or this
      "metadata": {...}
    }
  ]
}
```

**Options for "label":**
- `"valid"` - Image passes your validation criteria
- `"invalid"` - Image fails your validation criteria
- Leave blank to skip that sample

> **Tip:** At minimum, label 20-30 images to start (mix of valid and invalid for best results)

Once done, save as `training_manifest.json`

## Step 3: Train Your Model

Train on labeled images:

```bash
python train_model.py
```

**Output:**
- Shows number of labeled samples found
- Extracts features from images
- Trains Random Forest model
- Displays accuracy on train/test split
- Saves model to `models/xray_validator.pkl`

**Example output:**
```
✓ Found 30 labeled samples
  Valid: 15
  Invalid: 15
✓ Loaded 30 images
🤖 Training Random Forest model...
  Extracting features...
  Training...

📊 Training Results:
  Train accuracy: 0.9348
  Test accuracy: 0.8889

✓ Model saved to: models/xray_validator.pkl
```

## Step 4: Validate Images

Use your trained model to validate new images:

```bash
# Validate all images in a folder
python validate_images.py training_images/

# Validate a single image
python validate_images.py my_image.png
```

**Output:**
```
🔍 Validating 199 images...
======================================================================
✓ 101417395279.png             | VALID   | Confidence: 0.9234
✗ 143815546190.png             | INVALID | Confidence: 0.7456
✓ 166088960583.png             | VALID   | Confidence: 0.8912
...
======================================================================

📊 Summary:
  Valid: 142
  Invalid: 57
  Total: 199
  Valid ratio: 71.4%
```

## Understanding Your Validation Criteria

Before labeling, clarify: **What makes an X-ray image VALID?**

Examples:
- ✓ Presence of specific component/region
- ✓ Clean, artifact-free imaging
- ✓ Correct orientation/positioning
- ✓ Meets quality threshold
- ✓ No defects in specific area

Once you define this, use it to label your images consistently.

## Improving Model Accuracy

1. **More labels:** Label 50+ images for better training
2. **Balanced dataset:** Roughly equal valid/invalid samples
3. **Clear criteria:** Be consistent with your validation rules
4. **Document notes:** Add metadata to samples where validation is ambiguous

## Project Structure

```
xray-analysis/
├── training_images/              # Your 199 X-ray images
├── training_manifest_template.json  # Template (generated)
├── training_manifest.json           # Your labeled data (you create)
├── models/
│   └── xray_validator.pkl          # Trained model (generated)
├── prepare_training_data.py      # Step 1: Generate manifest
├── train_model.py                # Step 3: Train model
├── validate_images.py            # Step 4: Validate images
├── image_processor.py            # Image processing utilities
├── image_validator.py            # Validation approaches
├── data_loader.py                # Dataset management
└── README.md                     # Full documentation
```

## Troubleshooting

**"Manifest file not found"**
- Run `python prepare_training_data.py` first

**"No labeled samples found"**
- Make sure `training_manifest.json` has samples with `"label"` field filled in
- Field should be `"valid"` (lowercase)

**Low accuracy**
- Need more labeled samples (try 50+)
- Check label consistency (same rule applied throughout)
- Consider unlabeled/ambiguous images - skip those

**"Model not found"**
- Train first: `python train_model.py`

## Next Steps

After getting baseline accuracy:
- Retrain with more labels for better accuracy
- Explore `ensemble_validator.py` for combining multiple approaches
- Consider transfer learning CNN for even better results (100+ samples)

## Questions?

See full documentation in `README.md` for:
- All available validators (template matching, edge detection, etc.)
- Feature extraction details
- Dataset formats
- Extending the framework
