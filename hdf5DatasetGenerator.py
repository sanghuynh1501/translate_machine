# import the necessary packages
import numpy as np
import h5py

class HDF5DatasetGenerator:

    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
    
    def generator(self, passes=np.inf):
        # epochs = 0
        # while epochs < passes:
        for i in np.arange(0, self.numImages, self.batchSize):
            images1 = self.db["input1"][i: i + self.batchSize]
            images2 = self.db["input2"][i: i + self.batchSize]
            labels = self.db["labels"][i: i + self.batchSize]
            # if self.binarize:
            #     labels = np_utils.to_categorical(labels, self.classes)
            if self.preprocessors is not None:
                procImages1 = []
                for image in images1:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    procImages1.append(image)
                images1 = np.array(procImages1)
                procImages2 = []
                for image in images2:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    procImages2.append(image)
                images2 = np.array(procImages2)
            if self.aug is not None:
                (images1, images2, labels) = next(self.aug.flow(images1, images2, labels, batch_size=self.batchSize))
            yield (images1, images2, labels)
            # epochs += 1
    
    def close(self):
        self.db.close()

class HDF5DatasetGeneratorBert:

    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.batchSize = batchSize
        self.preprocessors = preprocessors
        self.aug = aug
        self.binarize = binarize
        self.classes = classes
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]
    
    def generator(self, passes=np.inf):
        # epochs = 0
        # while epochs < passes:
        for i in np.arange(0, self.numImages, self.batchSize):
            images1 = self.db["input1"][i: i + self.batchSize]
            images2 = self.db["input2"][i: i + self.batchSize]
            images3 = self.db["input3"][i: i + self.batchSize]
            labels = self.db["labels"][i: i + self.batchSize]
            # if self.binarize:
            #     labels = np_utils.to_categorical(labels, self.classes)
            if self.preprocessors is not None:
                procImages1 = []
                for image in images1:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    procImages1.append(image)
                images1 = np.array(procImages1)
                procImages2 = []
                for image in images2:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    procImages2.append(image)
                images2 = np.array(procImages2)
                procImages3 = []
                for image in images3:
                    for p in self.preprocessors:
                        image = p.preprocess(image)
                    procImages3.append(image)
                images3 = np.array(procImages3)
            if self.aug is not None:
                (images1, images2, images3, labels) = next(self.aug.flow(images1, images2, images3, labels, batch_size=self.batchSize))
            yield (images1, images2, images3, labels)
            # epochs += 1
    
    def close(self):
        self.db.close()