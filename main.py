from keras.callbacks import ModelCheckpoint

from model import *
from data import *
import cv2
import matplotlib.pyplot as plt

# 导入这两个文件中的所有函数

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# data_gen_args = dict(rotation_range=0,
#                      width_shift_range=0,
#                      height_shift_range=0,
#                      shear_range=0,
#                      zoom_range=0,
#                      horizontal_flip=False,
#                      fill_mode='nearest')  # 数据增强时的变换方式的字典

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
# 得到一个生成器，以batch=2的速率无限生成增强后的数据

model = unet()

model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
# 回调函数，第一个是保存模型路径，第二个是检测的值，检测Loss是使它最小，第三个是只保存在验证集上性能最好的模型

model.fit_generator(myGene, steps_per_epoch=200, epochs=30, callbacks=[model_checkpoint])
# steps_per_epoch指的是每个epoch有多少个batch_size，也就是训练集总样本数除以batch_size的值
# 上面一行是利用生成器进行batch_size数量的训练，样本和标签通过myGene传入

# ----------------------------------------------------------------------------------------------------------------------
# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)

results[results > 127] = 255
results[results < 127] = 0
# 返回值是：预测值的 Numpy 数组。

saveResult("data/membrane/test", results)

# ----------------------------------------------------------------------------------------------------------------------
# data_gen_args = dict(rotation_range=0.2,
#                      width_shift_range=0.05,
#                      height_shift_range=0.05,
#                      shear_range=0.05,
#                      zoom_range=0.05,
#                      horizontal_flip=True,
#                      fill_mode='nearest')
# myGene = trainGenerator(2, 'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)
# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss', verbose=1, save_best_only=True)
# model.fit_generator(myGene, steps_per_epoch=30, epochs=1, callbacks=[model_checkpoint])

# ----------------------------------------------------------------------------------------------------------------------
# testGene = testGenerator("data/membrane/test")
# model = unet()
# model.load_weights("unet_membrane.hdf5")
# results = model.predict_generator(testGene, 104, verbose=1)
# saveResult("data/membrane/test", results)
# ----------------------------------------------------------------------------------------------------------------------
