#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import lockon_eval as eval
import json
import time
import urllib
import falcon
import datetime
import cv2
import os
import numpy as np
from boto.s3.key import Key
from boto.s3.connection import S3Connection

""" メインモジュール """

# OpenCVのデフォルトの顔の分類器のpath
cascade_path = './haarcascades/haarcascade_frontalface_default.xml'
eyes_cascade_path = './haarcascades/haarcascade_eye.xml'

# アップロード先のbase url
s3_base_path = "https://s3-ap-northeast-1.amazonaws.com/lockon/"
date = datetime.date.today().strftime("%Y%m%d")
color = (0, 0, 255) #赤

class LockonResource(object):
  """ post 画像ファイルを取得しスコアのもっとも近い社員と認識画像のパスを返す """
  def on_post(self, req, resp):
    body = req.stream.read()
    req_body = json.loads(body.decode('utf-8'))
    image_path= req_body['image_path']
    
    face,path = main(image_path, './model2.ckpt')
    msg = {"image": path, "result": face}
    print(msg)
    resp.body = json.dumps(msg)
class LockonHealthCheck(object):
  def on_get(self, req, resp):
  resp.body = "pong"

app = falcon.API()
app.add_route("/lockon", LockonResource()) 
app.add_route("/api/ping", LockonHealthCheck())

def url_to_image(url):
	""" ネット上の画像ファイルをcv2のimageへ変換 """
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return image

def s3_upload(file_path, image):
  """ 画像をS3へアップロード """
  # 一旦ローカルへ画像保存
  cv2.imwrite(file_path, image)
  # S3接続
  conn = S3Connection()
  bucket  =conn.get_bucket('lockon')
  # ファイル名の指定
  key_name = date + '/' + file_path
  k = Key(bucket)
  k.key = key_name
  # アップロード
  k.set_contents_from_filename(file_path)
  k.make_public()
  # アップロード完了したら消す
  os.remove(file_path)

  return s3_base_path + key_name

# 指定した画像(img_path)を学習結果(ckpt_path)を用いて判定する
def main(img_path, ckpt_path):
  image = url_to_image(img_path)
  if image is None:
    return {message : "not open file"}
    print('Not open : ',line)

  cascade = cv2.CascadeClassifier(cascade_path)
  facerect = cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

  if len(facerect) > 0:
    faces =[]
    for rect in facerect:
      
      # 顔だけ切り出して保存
      x = rect[0]
      y = rect[1]
      width = rect[2]
      height = rect[3]
      face_img = image[y:y + height, x:x + width]

      # 顔検出
      # open cvで顔と判断されたもので目が2つあるもののみ対象とする
      eyes = []
      eye_cascade = cv2.CascadeClassifier(eye_cascade)
      for eye in eye_cascade.detectMultiScale(face_img):
        # 始点の高さが元画像の下半分にあるようならおそらくそれは誤検出
        if eye[1] > face_img.shape[0] / 2:
          continue
        eyes.append(eye)
      # 目が2つ以外の場合は顔としない
      if len(eyes) != 2:
          print('{} eyes found...'.format(len(eyes)))
          continue

      rank, img = eval.evaluation(face_img, ckpt_path)
      # 顔と判別された部分のみ囲む
      cv2.rectangle(image, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=10)
      # 判定結果と加工した画像のpathを返す
      faces.append([rank, img][0][0])

    # S３に画像を保存して、パスを返す
    file_path = "loackon_" + date + str(int(time.time() * 1000)) + ".jpg"
    s3_path = s3_upload(file_path, image)
    return faces, s3_path
  
  # 画像内に顔が認識されなかった場合は何
  return "none",""
