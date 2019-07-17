import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, input1_dims, input2_dims, label_dims, outputPath, dataKey1="input1", dataKey2="input2", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)
        self.db = h5py.File(outputPath, "w")
        self.data1 = self.db.create_dataset(dataKey1, input1_dims, dtype="float")
        self.data2 = self.db.create_dataset(dataKey2, input2_dims, dtype="float")
        self.labels = self.db.create_dataset("labels", label_dims, dtype="float")
        self.bufSize = bufSize
        self.buffer = {"data1": [], "data2": [], "labels": []}
        self.idx = 0
    
    def add(self, rows1, rows2, labels):
        self.buffer["data1"].extend(rows1)
        self.buffer["data2"].extend(rows2)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["data1"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data1"])
        self.data1[self.idx:i] = self.buffer["data1"]
        self.data2[self.idx:i] = self.buffer["data2"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data1": [], "data2": [], "labels": []}
    
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels
    
    def close(self):
        if len(self.buffer["data1"]) > 0:
            self.flush()
        self.db.close()

class HDF5DatasetWriterBert:
    def __init__(self, input1_dims, input2_dims, input3_dims, label_dims, outputPath, dataKey1="input1", dataKey2="input2", dataKey3="input3", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The supplied ‘outputPath‘ already exists and cannot be overwritten. Manually delete the file before continuing.", outputPath)
        self.db = h5py.File(outputPath, "w")
        self.data1 = self.db.create_dataset(dataKey1, input1_dims, dtype="float")
        self.data2 = self.db.create_dataset(dataKey2, input2_dims, dtype="float")
        self.data3 = self.db.create_dataset(dataKey3, input3_dims, dtype="float")
        self.labels = self.db.create_dataset("labels", label_dims, dtype="float")
        self.bufSize = bufSize
        self.buffer = {"data1": [], "data2": [], "data3": [], "labels": []}
        self.idx = 0
    
    def add(self, rows1, rows2, rows3, labels):
        self.buffer["data1"].extend(rows1)
        self.buffer["data2"].extend(rows2)
        self.buffer["data3"].extend(rows3)
        self.buffer["labels"].extend(labels)
        if len(self.buffer["data1"]) >= self.bufSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buffer["data1"])
        self.data1[self.idx:i] = self.buffer["data1"]
        self.data2[self.idx:i] = self.buffer["data2"]
        self.data3[self.idx:i] = self.buffer["data3"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data1": [], "data2": [], "data3": [], "labels": []}
    
    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels
    
    def close(self):
        if len(self.buffer["data1"]) > 0:
            self.flush()
        self.db.close()