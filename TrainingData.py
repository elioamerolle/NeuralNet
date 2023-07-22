from tkinter import DoubleVar

# TrainingData.py

"""
The class to access and use the training data
"""
class TrainingData: 
    # this is the class to access the training data
    
    """
    The initializer for the class
    """
    def __init__(self):
        # the initializer for the class
        
        self.parse(self.getText())
        # parses the text
    
    """
    A function to retrieve the real data
    :returns text: the real text data
    """
    def getText(self):
        # A function to retrieve the real data
        
        dataSetFile = open("4x4-image-dataset-percentage.txt", "r")
        # gets the data set file
        
        text = dataSetFile.read()
        # gets the text of the data set
        
        return text
        # returns the text
    
    """
    A function to parse the text
    :param text: the inputted text
    """
    def parse(self, text):
        # A function to parse the text
        
        self.initialData = text.split("\n")
        # stores the first part of the data by splitting each image
        
        self.images = []
        # creates the image data
        
        self.expectedOutputs = []
        # creates the image data
        
        for initialData in self.initialData:
            # loops through each image string
            
            splittedData = initialData.split(",")
            # splits the data
            
            self.expectedOutputs.append([float(splittedData[1])])
            # adds to the expected output the float
            
            imageData = []
            # creates the image data to store
            
            for binaryCharacter in splittedData[0]:
                # loops through each character
                
                value = 0
                # the value to add to the image
                
                if binaryCharacter == '1':
                    # if the character is a 1
                    
                    value = 1
                    # sets the value to 1
                
                imageData.append(value)
                # adds the binary character
            
            self.images.append(imageData)
            # adds to the expected output the float
    
    """
    A function to tests the class
    """
    def test(self):
        # tests the class
        
        print("TrainingData.test: Testing the fake training data")
        
        print("TrainingData.test: [WARNING] When testing the data the real training data is not being used,\nif you want to use it after testing it re-initialize it like the following:\nTrainData.__init__()")
        
        print("TrainingData.test: Parsing training data")
        
        self.parse("00,0\n01,1\n10,1\n11,0")
        # parses the data
        
        print("TrainingData.test: Succeeded Parsing Data")
        print("TrainingData.test: Printing Variables")
        
        print(self.initialData)
        print(self.expectedOutputs)
        print(self.images)
        
        print("TrainingData.test: Testing retreiving functions")
        
        print(self.getImage(0))
        print(self.getExpectedOutput(0))
        
        print("TrainingData.test: Succeeded retreiving functions")
        
        print("TrainingData.test: Failing test retreiving functions")
        
        print(self.getImage(200))
        print(self.getExpectedOutput(190))
        
        print("TrainingData.test: Succeeded failing test retreiving functions")
        
        
        print("TrainingData.test: Successfully Completed Tests")
    
    """
    A function to retreive an image
    :param index: a number for the index of the image
    :returns image: the 16 size array of the image
    """
    def getImage(self, index):
        # a function to retreive an image (returns a 16 size array of 1 or 0)
        
        if index >= len(self.images):
            # if the index is greater than the amount of image data
            
            print("TrainingData.getImage: Error, unable to retreive image due to index being out of bounds. Index: " + str(index) + ", Length: " + str(len(self.images)))
            
            return [0] * 16
            # returns an empty image
        
        return self.images[index]
        # returns the image
    
    """
    A function to retreive the expected output
    :param index: a number for the index of the image
    :returns percent: the percent float from 0 to 1
    """
    def getExpectedOutput(self, index):
        # a function to retreive the expected output (returns a float number from 0 to 1)
        
        if index >= len(self.expectedOutputs):
            # if the index is greater than the amount of image data
            
            print("TrainingData.getExpectedOutput: Error, unable to retreive expected output due to index being out of bounds. Index: " + str(index) + ", Length: " + str(len(self.expectedOutputs)))
            
            return 0
            # returns an empty image
        
        return self.expectedOutputs[index]
        # returns the expected output
