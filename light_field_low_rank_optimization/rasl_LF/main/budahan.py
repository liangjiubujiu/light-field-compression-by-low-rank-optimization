# -*-coding:utf-8 -*-
"""
@ author :vivian

Created on 2018/5/24 14:36

import os
import rasl

if __name__ == "__main__":

    digits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data/Digits_3")
    rasl.application.demo_cmd(
        description="Align handwritten digits using RASL",
        path=digits_dir, frame=0, tform=rasl.EuclideanTransform)

"""

import os
import rasl

if __name__ == "__main__":

    digits_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "data/data_test/Image")
    rasl.application.demo_cmd(
        description="Align handwritten digits using RASL",
        path=digits_dir, frame=0, tform=rasl.EuclideanTransform)
#将图片读入的时候增加cv2.resize(im,[200,200])