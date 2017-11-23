import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

from model import  alexnet
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy


from datasets.nuclei import NucleiLoader
logger=logging.getLogger(__name__)

parser=argparse.ArgumentParser()
parser.add_argument("--ewc",action='store_true')
parser.add_argument("-e",type=int)
args=parser.parse_args()
print(args)
EPOCHS=args.e

if args.ewc:
    logger.critical("Running with EWC")
else:
    logger.critical("Running without EWC (Vanilla loss)")
    

import torch
torch.manual_seed(1234)


class EWCTrainer:

    def __init__(self,classes,sample_data,batch_size=100):

        self.sample_data = sample_data
        self.batch_size = batch_size
        self.model = alexnet.AlexNet(classes);



    def batchify(self,X,y):
        i=0
        while 1:
            batch_X,batch_y=X[i:i+self.batch_size],y[i:i+self.batch_size]
            if batch_X.shape[0]!=self.batch_size:
                i=0
                difference=self.batch_size-batch_X.shape[0]
                batch_X=numpy.vstack((batch_X,X[i:i+difference]))
                batch_y=numpy.hstack((batch_y,y[i:i+difference]))
            yield batch_X,batch_y
            i+=self.batch_size


    def get_accuracy(self,output,label):
        total=output.shape[0]
        correct=output[output==label].shape[0]
        accuracy=correct/total
        logger.info("Accuracy: {}".format(accuracy))
        return accuracy

    def train_on_task(self,name,batch_generator,EPOCHS=EPOCHS):
        logger.info("Training task {}".format(name))
        for i in range(EPOCHS):
            train_x,train_y=next(batch_generator)
            train_x = train_x
            train_x = train_x.permute(0, 3, 1,2)
            #train_x=numpy.repeat(train_x[:,:,:,numpy.newaxis],3,axis=3)
            print(train_x.shape)
            print(train_y.shape)
            self.model.partial_fit(train_x,train_y)
            if i==EPOCHS-1:
                logger.info("Accuracy on {}: {}".format(name,get_accuracy(model.predict(train_x),train_y)))
        return train_x,train_y

    def partial_fit(self,data_x1,data_y1,data_x2,data_y2):
        first_data=self.train_on_task("MNIST",self.batchify(data_x1,data_y1))
        if args.ewc:
            self.model.consolidate(numpy.transpose(self.sample_data[0][:,:,:,numpy.newaxis],(0,3,1,2)),self.sample_data[1])
        second_data=self.train_on_task("FASHION",self.batchify(data_x2,data_y2))
        logger.info("Task B is trained. Now computing accuracy for task A to check forgetting.")
        woc=self.get_accuracy(self.model.predict(first_data[0]),second_data[1])
        return woc;

data = NucleiLoader();
model = None;
for x, y in data.load_train(0,batch_size=100):

    if  model is  None:
        sample_data= x,y
        model = EWCTrainer(2,sample_data)
    model.partial_fit(x,y,x,y)


