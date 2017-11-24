import argparse
#sys.path.append("/home/adityas/Projects/DataScience2/model/")

from model import  alexnet
import logging
logging.basicConfig(level=logging.DEBUG)

from keras.datasets import mnist
import numpy
from scores import Score

from datasets.nuclei import NucleiLoader
from datasets.epi import EpitheliumLoader 
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

    def __init__(self,classes,batch_size=100):

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




    def train_on_task(self,name,train_x,train_y,EPOCHS=EPOCHS):
        logger.info("Training task {}".format(name))
        train_x = train_x.permute(0, 3, 1,2)
        loss=self.model.partial_fit(train_x,train_y)
        logger.debug("loss: {}".format(loss))
        return train_x,train_y

    def train_on_task_consolidate(self,name,train_x,train_y,EPOCHS=EPOCHS):
        logger.info("Training task {}".format(name))
        train_x = train_x.permute(0, 3, 1,2)
        self.model.consolidate(train_x,train_y)

    def predict(self,test_x):
        test_x = test_x.permute(0, 3, 1,2)
        return  self.model.predict(test_x)


    def partial_fit(self,data_x1,data_y1):
        first_data=self.train_on_task("MNIST",data_x1,data_y1)
        logger.info("Task is trained ")
        #woc=self.get_accuracy(self.model.predict(first_data[0]),data_y1)

data_nuclei = NucleiLoader()
data_epi= EpitheliumLoader()
trainer = EWCTrainer(2)

y1 = list();
y2 = list();
y_orig = list();

#training task -1 without ewc
for i in range(0,EPOCHS):
    for x, y in data_nuclei.load_train(0,batch_size=100):
        print("on main  funcition")
        print(x.shape)
        print(y.shape)
        sample_data = x.clone(),y.clone()
        trainer.train_on_task("TASK MAIN WITHOUT EWC",x.clone(),y.clone())
        

for x,y in data_nuclei.load_test(0,batch_size=100):
    y1.extend(trainer.predict(x.clone()))
    y_orig.extend(y.numpy())
    

print("----------------------Now trianing with ewc and getting metrics ---------------------------------------->")
trainer = EWCTrainer(2)


for i in range(0,EPOCHS):
    for x, y in data_epi.load_train(0,batch_size=100):
        trainer.train_on_task("Task1 -AD HOC TASK",x,y)
        sample_data = x.clone(),y.clone()
        

trainer.train_on_task_consolidate("CONSLIDATE WEIGHTS",sample_data[0],sample_data[1])

for i in range(0,EPOCHS):
    for x, y in data_nuclei.load_train(0,batch_size=100):
        print("on main  funcition")
        print(x.shape)
        print(y.shape)
        sample_data = x.clone(),y.clone()
        trainer.train_on_task("TASK-MAIN EWC TESTING TASK",x.clone(),y.clone())
        

print("recording prediction of EWC Testing task")

for x,y in data_nuclei.load_test(0,batch_size=100):
    y2.extend(trainer.predict(x.clone()))

  

s = Score();
print("<---------------------------SCORE METRICS ----------------------------------------------------->")
print(s.build_table(y_orig,y1,y2));
s.plot_roc(y_orig,y1,y2)
s.plot_loss(y_orig,y1)

print("--------------------------SCORE METRIC END ----------------")







