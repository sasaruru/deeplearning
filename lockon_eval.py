#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import cv2
import tensorflow as tf
import random
import time
import lockon_model as model

# 識別ラベルと各ラベル番号に対応する名前
HUMAN_NAMES = {
  0: u"阿部",
  1: u"荒",
  2: u"遠藤",
  3: u"深澤",
  4: u"伏見",
  5: u"原田",
  6: u"原口",
  7: u"ひろし",
  8: u"コジマ",
  9: u"近藤",
  10: u"前川",
  11: u"三浦",
  12: u"三宅",
  13: u"宮崎",
  14: u"百瀬",
  15: u"仁田脇",
  16: u"大堀",
  17: u"大城",
  18: u"さいき",
  19: u"坂本",
  20: u"佐々木",
  21: u"しんち",
  22: u"そむ",
  23: u"高橋",
  24: u"タスク",
  25: u"和田",
  26: u"山田",
  27: u"ゆーた"
}

def evaluation(face_img, ckpt_path):
  """ 識別した結果を返す """
  tf.reset_default_graph()
  # データを入れる配列
  image = []
  #img = cv2.imread(img_path)
  img = cv2.resize(face_img, (28, 28))
  # 画像情報を一列にした後、0-1のfloat値にする
  image.append(img.flatten().astype(np.float32)/255.0)
  # numpy形式に変換し、TensorFlowで処理できるようにする
  image = np.asarray(image)
  # 入力画像に対して、各ラベルの確率を出力して返す(main.pyより呼び出し)
  logits = model.inference_deep(image, 1.0, 28, 28)
  # We can just use 'c.eval()' without passing 'sess'
  sess = tf.InteractiveSession()
  # restore(パラメーター読み込み)の準備
  saver = tf.train.Saver()
  # 変数の初期化
  sess.run(tf.global_variables_initializer())
  if ckpt_path:
    # 学習後のパラメーターの読み込み
    saver.restore(sess, ckpt_path)
  # sess.run(logits)と同じ
  softmax = logits.eval()
  # 判定結果
  result = softmax[0]
  # 判定結果を%にして四捨五入
  rates = [round(n * 100.0, 1) for n in result]
  humans = []
  # ラベル番号、名前、パーセンテージのHashを作成
  for index, rate in enumerate(rates):
    name = HUMAN_NAMES[index]
    humans.append({
      'label': index,
      'name': name,
      'rate': rate
    })
  # パーセンテージの高い順にソート
  rank = sorted(humans, key=lambda x: x['rate'], reverse=True)
  return rank, img
