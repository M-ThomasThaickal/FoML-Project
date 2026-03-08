from datasets import load_dataset, concatenate_datasets

# Load in data and combine "train" and "test" into one big dataset
dataset = load_dataset("tanganke/stanford_cars")
# totaL: 16,185 images
full = concatenate_datasets([dataset["train"], dataset["test"]]) 

# Take splits 
split1 = full.train_test_split(test_size=0.30, seed=42)
split2 = split1["test"].train_test_split(test_size=0.50, seed=42)

train_ds = split1["train"]  # ~11,330
val_ds   = split2["train"]  # ~2,428
test_ds  = split2["test"]   # ~2,428

