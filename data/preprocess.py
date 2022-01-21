import os
import glob



import sys 
# sys.path.append("..")
import cfg
import random



if __name__ == '__main__':
    traindata_path = r"D:\uu\pytorch_classification-master\data\data_uu"
    labels = os.listdir(traindata_path)
    valdata_path = r"D:\uu\pytorch_classification-master\data\data_uu"
    ##写train.txt文件
    txtpath = r"D:\uu\pytorch_classification-master\processed-data\uu_14"
    print(labels)
    for index, label in enumerate(labels):
        imglist = glob.glob(os.path.join(traindata_path,label, '*.jpg'))
        # print(imglist)
        random.shuffle(imglist)
        print(len(imglist))
        # trainlist = imglist[:int(1*len(imglist))]
        vallist = imglist[(int(0.8*len(imglist))+1):]
        # with open(txtpath + '\\oil_train.txt', 'a')as f:
        #     for img in trainlist:
        #         # print(img + ' ' + str(index))
        #         f.write(img + ' ' + str(index))
        #         f.write('\n')

        with open(txtpath + '\\oil_val.txt', 'a')as f:
            for img in vallist:
                # print(img + ' ' + str(index))
                f.write(img + ' ' + str(index))
                f.write('\n')


    # imglist = glob.glob(os.path.join(valdata_path, '*.png'))
    # with open(txtpath + '\\oil_test.txt', 'a')as f:
    #     for img in imglist:
    #         f.write(img)
    #         f.write('\n')


