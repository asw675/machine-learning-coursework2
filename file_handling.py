import scipy.io as scio
import os
import shutil

# load the test and train classification list
test_path = "../test_list.mat"
train_path = "../train_list.mat"

# images path (untar by images.tar)
path_img = "../images/"
if not os.path.exists(path_img+"test"):
    os.makedirs(path_img+"test")
if not os.path.exists(path_img+"train"):
    os.makedirs(path_img+"train")

test_data = scio.loadmat(test_path)

# create test folder
for t in test_data["labels"]:
    if not os.path.exists(path_img+"test/"+str(t[0])):
        os.makedirs(path_img+"test/"+str(t[0]))
train_data = scio.loadmat(train_path)

# create train folder
for t in train_data["labels"]:
    if not os.path.exists(path_img+"train/"+str(t[0])):
        os.makedirs(path_img+"train/"+str(t[0]))

# classify the test images to test folder
for t in range(len(test_data["file_list"])):
    source_path = ("../images/Images/"+test_data["file_list"][t][0][0])
    target_path = (path_img+"test/"+str(test_data["labels"][t][0])+"/"+test_data["file_list"][t][0][0].split("/", 1)[1])
    shutil.copyfile(source_path,target_path)

# classify the train images to test folder
for t in range(len(train_data["file_list"])):
    source_path = ("../images/Images/"+train_data["file_list"][t][0][0])
    target_path = (path_img+"train/"+str(train_data["labels"][t][0])+"/"+train_data["file_list"][t][0][0].split("/", 1)[1])
    shutil.copyfile(source_path,target_path)