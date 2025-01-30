import splitfolders
import os

def split_data():
    """
    Splits the Fruit and Vegetable Diseases dataset into Train, Validation, and Test folders.
    """
    # Define the absolute or relative path to the dataset
    dataset_path = os.path.abspath("../data/Fruit And Vegetable Diseases Dataset")  # Adjust path as needed
    output_path = os.path.abspath("../data/split_dataset")  # Adjust output path as needed

    # Split the dataset into Train (70%), Validation (15%), and Test (15%)
    splitfolders.ratio(
        input=dataset_path,  # Corrected parameter
        output=output_path,
        seed=42,             # For reproducibility
        ratio=(0.7, 0.15, 0.15),
        group_prefix=None,   # Prevent grouping (treat each class independently)
        move=False           # Copy files instead of moving
    )

    print(f"Dataset successfully split into: {output_path}")
