import pandas as pd
from ml.config import config
import os
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

train_data = pd.read_csv(config.train_path)
test_data = pd.read_csv(config.test_path)
val_data=train_data.sample(frac=0.25)
train_set=train_data.drop(val_data.index)


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# columns = train_data.columns
#
# labels = train_data['Label'].unique()
#
# for i in labels:
#     create_folder(os.path.join(config.train_dir, str(i)))
#
# for image in list(train_set['Data']):
#     label = train_set.loc[train_set['Data'] ==image, 'Label'].iloc[0]
#     des_path = os.path.join(config.train_dir ,str(label), image)
#
#     src_path =os.path.join(config.train_set, image)
#
#     copyfile(src_path, des_path)

height =256
width = 256
train_batch_size =32

augmented_set = datagen.flow_from_directory(
        config.train_dir,
        target_size=(height,width),
        color_mode = 'rgb',
        class_mode="categorical",
        batch_size=train_batch_size,
        shuffle=True,
        )

X, y = augmented_set.next()

model = VGG16(include_top=False)

print(X.shape)
print(y.shape)


