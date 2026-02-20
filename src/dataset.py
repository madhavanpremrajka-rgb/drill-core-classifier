import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore

def which_dataset(C, R, L):
    """
    Gets the directory for the dataset given the conditions
    
    :param C: Number of classes: 7 or 35
    :param R: Resolution of the images: 32 or 128
    :param L: Augmentation level: 0.0, 0.1, 0.2, 0.3, 0.4
    """

    assert C in [7, 35], f"Expected C to be either 7 or 35, got {C}"
    assert R in [32, 128], f"Expected R to be either 32 or 128, got {R}"
    assert L in [0.0, 0.1, 0.2, 0.3, 0.4], f"Expected L to be either 0.0, 0.1, 0.2, 0.3, or 0.4, got {L}"

    root_dir = "../Data_Processed"
    dataset_name = f"Dataset-{C}_R-{R}_L-{L}_I-T"
    
    return os.path.join(root_dir, dataset_name)
    
def generate_dataset(C, R, L, batch_size=64, seed=26):
    """
    Loads train, val, and test splits for a given dataset configuration.    

    :param C: Number of classes: 7 or 35
    :param R: Resolution of the images: 32 or 128
    :param L: Augmentation level: 0.0, 0.1, 0.2, 0.3, 0.4
    :param batch_size: Number of images per batch (default 64)
    :param seed: seed (default 26)
    :return: Tuple of (train_data, valid_data, test_data)
    """
    datagen = ImageDataGenerator(rescale=1/255.)

    dataset_dir = which_dataset(C, R, L)
    train = 'train'
    valid = 'val'
    test = 'test'

    train_dir = os.path.join(dataset_dir, train)
    valid_dir = os.path.join(dataset_dir, valid)
    test_dir = os.path.join(dataset_dir, test)

    train_data = datagen.flow_from_directory(
        train_dir,
        batch_size = batch_size,
        target_size = (R, R),
        class_mode = 'categorical',
        seed = seed,
        shuffle = True
    )
    valid_data = datagen.flow_from_directory(
        valid_dir,
        batch_size = batch_size,
        target_size = (R, R),
        class_mode = 'categorical',
        seed = seed
    )
    test_data = datagen.flow_from_directory(
        test_dir,
        batch_size = batch_size,
        target_size = (R, R),
        class_mode = 'categorical',
        seed = seed
    )

    return train_data, valid_data, test_data