import os
import pandas as pd
import shutil

isExists1 = os.path.exists("E:\\中石油工作\\class14_resnet_picture")
isExists2 = os.path.exists("E:\\中石油工作\\class14_resnet_picture\\0")
if isExists1 == False:
    os.makedirs("E:\\中石油工作\\class14_resnet_picture")
if isExists2 == False:
    for i in range(14):
        os.makedirs("E:\\中石油工作\\class14_resnet_picture\\" + str(i))

# 
def main(list):
    lables_list = set()
    img_absolute_path, lable = i[0].split( )[0], i[1]
    img_relatively_path = i[0].split('\\')[-1].split( )[0]
    lables_list.add(lable)
    return img_absolute_path, lable, img_relatively_path, lables_list

text_file = open("D:\\中石油工作\\pytorch_classification-master\\result\\resnext101_32x32d_submission_copy_oil14_eff.csv", "rb")
list = pd.read_csv(text_file)
# print(list)
for i in list.values.tolist():
    img_absolute_path, lable, img_relatively_path, lables_list = main(i)
    for j in lables_list:
        if int(lable) == j:
            # print(img_absolute_path)
            # print("E:\\中石油工作\\class13_picture\\" + str(lable) + "\\" + img_relatively_path)
            shutil.move(img_absolute_path, "E:\\中石油工作\\class14_resnet_picture\\" + str(lable) + "\\" + img_relatively_path)


