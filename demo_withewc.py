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
from EWCTrainer import EWCTrainer
logger=logging.getLogger(__name__)

parser=argparse.ArgumentParser()
parser.add_argument("--ewc",action='store_true')
parser.add_argument("-e",type=int)
args=parser.parse_args()
print(args)
EPOCHS=args.e


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
        break;
        

for x,y in data_nuclei.load_test(0,batch_size=100):
    y.extend(trainer.predict(x.clone()))
    y_orig.extend(y.numpy())
    break;
    


s = Score();
print("<---------------------------SCORE METRICS ----------------------------------------------------->")
print(s.build_table(y_orig,y));
s.plot_roc(y_orig,y)
s.plot_loss(y_orig,y)

print("--------------------------SCORE METRIC END ----------------")





