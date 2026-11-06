# Plant pipeline showcase

`plant_pipeline_showcase.ipynb` shows the drone-image canopy pipeline on one field and two flight dates. The notebook shows these parts:

- Grid and polygons
- Alignment across dates
- Segmentation
- Area measurement
- Visual inspection, with difficult cases

`plant_pipeline_showcase.html` is the same notebook with all outputs. You can read this file and not run the notebook.

## Run the notebook

1. Get the repository. It contains the `examples/S_2021/` data.
2. Get the 06_28 and 07_03 images from the Releases page of this repository.
3. Put the images in `examples/S_2021_crops/`.
4. Run these commands from the top folder of the repository:

```sh
cd showcase
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace plant_pipeline_showcase.ipynb
```

## Files

- The `Step*.py` modules of the repository contain the measurement code.
- `helpers.py` contains the plot and load functions of the notebook.
- `showcase_config.toml` selects the field and the dates.

The notebook writes all its files to `showcase/outputs/`. It does not change `examples/`.
