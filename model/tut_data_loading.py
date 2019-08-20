import matplotlib
matplotlib.use('tkagg', warn=False)
import warnings
import matplotlib.pyplot        as plt
import pandas                   as pd
import numpy                    as np
import torchvision.transforms   as trf
from torch.utils.data           import Dataset
from pathlib                    import Path


def show_img_with_landmarks(image, landmarks):
    fig = plt.figure()
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    ax = plt.gca()
    ax.axis('off')


class FaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.landmarks_frame    = pd.read_csv(csv_file)
        self.image_folder       = Path(image_folder)
        self.transform          = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        self.image_name = self.image_folder / self.landmarks_frame.iloc[idx, 0]

        self.image      = plt.imread(self.image_name)
        self.landmarks  = self.landmarks_frame.iloc[idx, 1:].values
        self.landmarks  = np.reshape(self.landmarks, (-1,2))
        self.sample     = {'image': self.image,
                           'landmarks': self.landmarks}
        if self.transform:
            self.sample = self.transform(self.sample)
        
        return self.sample


class Rescale():
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        
        self.output_size = output_size

    def __call__(self, sample):
        self.image, self.landmarks = sample['image'], sample['landmarks']

        height, width = self.image.shape[:2]
        if isinstance(self.output_size, int):
            if height > width:
                newHeight, newWidth = self.output_size * height / width, self.output_size
            else:
                newHeight, newWidth = self.output_size, self.output_size * height / width
        else:
            newHeight, newWidth = self.output_size
        
        newHeight, newWidth = int(newHeight), int(newWidth)

        self.image = trf.Resize(self.image, (newHeight, newWidth))

        return {'image': self.image, 'landmarks': self.landmarks}


class RandomCrop():
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        self.image, self.landmarks = sample['image'], sample['landmarks']

        h, w = self.image.shape[:2]

        newH, newW = self.output_size

        top     = np.random.randint(0, h - newH)
        left    = np.random.randint(0, w - newW)

        self.image = self.image[top: top + newH,
                                left: left + newW]
        self.landmarks = self.landmarks + [left, top]

        return {'image': self.image, 'landmarks': self.landmarks}


datasetFolder = Path("../datasets/torch/face_poses_tut/data/faces/")
landmarks_frame = pd.read_csv(datasetFolder / "face_landmarks.csv")

csvPath = datasetFolder / "face_landmarks.csv"
dataset = FaceLandmarksDataset(csvPath, datasetFolder)

print("Sample | Image         | Landmarks")
for i in range(len(dataset)):
    sample = dataset[i]

    print(str(i).ljust(5), str(sample['image'].shape).rjust(16), str(sample['landmarks'].shape).rjust(10))

    show_img_with_landmarks(**sample)
    ax = plt.gca()
    ax.set_title("Sample {}".format(i))
    
    plt.show()
