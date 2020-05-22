import os
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

train_path = os.path.join(BASE_PATH, "ml", "dataset", "Train", "trainset.csv")
test_path = os.path.join(BASE_PATH, "ml", "dataset", "Test", "testset.csv")
train_dir = os.path.join(BASE_PATH, "ml", "dataset", "Train", "Train_dir")
train_set = os.path.join(BASE_PATH, "ml", "dataset", "Train", "trainset")