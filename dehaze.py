# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse


def dark_channel(img, size = 15):                   #暗通道的计算
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))             #获取RGB三个通道最小值，生成单通道像素值最小的灰度图
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))  #构造了一个15*15的方形范围，为图像腐蚀提供卷积核范围
    dc_img = cv2.erode(min_img,kernel)              #用15*15的方框进行图像腐蚀操作，生成暗通道图像
    return dc_img                                   #返回暗通道图


#估算全局大气光值
def get_atmo(img, percent = 0.001):                                         
    """
    1.计算有雾图像的暗通道
    2.用一个Node的结构记录暗通道图像每个像素的位置和大小，放入list中
    3.对list进行降序排序 ？？？
    4.按暗通道亮度前0.1%(用percent参数指定百分比)的位置，在原始有雾图像中查找最大光强值
    """ 
    mean_perpix = np.mean(img, axis = 2).reshape(-1)                        #取rgb的均值，生成一个灰度图；将灰度图像素矩阵转换成列表list[像素值].
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]  #读取list前0.1%的像素值，并返回此片段
    return np.mean(mean_topper)                                             #返回前0.1%的均值 即A


#估算传输函数
def get_trans(img, atom, w = 0.95):         
    """
    w为去雾程度，一般取0.95
    w的值越小，去雾效果越不明显
    """
    x = img / atom                          
    t = 1 - w * dark_channel(x, 15)         #文章中固定的暗通道计算方块即15*15
    return t                                #返回图片的传射图                           


#导向滤波器
def guided_filter(p, i, r, e):                          
    """
    :param p: input image           #输入图片
    :param i: guidance image        #导图
    :param r: radius                #半径
    :param e: regularization        #规整化
    :return: filtering output q     #滤波后的输出图片
    """
    #1 
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))       #导向均值
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))       #原始均值
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))   #自相关均值
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))  #互相关均值
    #2
    var_I = corr_I - mean_I * mean_I                    #自相关方差var
    cov_Ip = corr_Ip - mean_I * mean_p                  #互相关方差cov
    #3
    a = cov_Ip / (var_I + e)                            #计算窗口线性变换参数系数
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))       #计算系数a、b均值
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b                             #生成引导滤波输出矩阵
    return q


#去雾主程序
def dehaze(path, output = None):                                    
    im = cv2.imread(path)

    img = im.astype('float64') / 255                                    #压缩RGB通道值于0到1
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atom = get_atmo(img)                                                #全局大气光值A，为传输函数计算传参
    trans = get_trans(img, atom)                                        #计算传输函数
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)           #用灰度图作为导图，平滑透射图，优化传射图
    trans_guided = cv2.max(trans_guided, 0.25)
    """
    1.t0 最小透射率值，一般取0.25
    2.透射图t 的值过小--->图像会整体向白场过度
    3.因此一般设置一阈值t0：当t值小于t0时，令t=t0=0.25
    """

    result = np.empty_like(img)                                         #创建图像大小矩阵
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom   #生成去雾图像

    cv2.imshow("source",img)                                            #显示、保存结果
    cv2.imshow("result", result)
    cv2.waitKey()
    if output is not None:
        cv2.imwrite(output, result * 255)

# 通过命令行传递参数
parser = argparse.ArgumentParser()      #创建一个解析对象 方法参数须知：一般我们只选择用description
parser.add_argument('-i', '--input')    #向该对象中添加你要关注的命令行参数和选项
parser.add_argument('-o', '--output')
args = parser.parse_args()              #进行解析


if __name__ == '__main__':
    if args.input is None:
        dehaze('image/canon3.bmp')
    else:
        dehaze(args.input, args.output)