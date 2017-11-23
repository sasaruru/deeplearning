#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

if __name__ == '__main__':
  outdir = sys.argv[1]

  if not os.path.isdir(outdir):
    sys.exit('%s is not directory' % outdir)

  names = {
    "abe": 0,
    "ara": 1,
    "endo": 2,
    "fukasawa": 3,
    "fushimi": 4,
    "harada":5,
    "haraguchi": 6,
    "itou": 7,
    "kojima": 8,
    "kondo": 9,
    "maekawa": 10,
    "miura": 11,
    "miyake": 12,
    "miyazaki": 13,
    "momose": 14,
    "nitawaki": 15,
    "ohori": 16,
    "oshiro": 17,
    "saiki": 18,
    "sakamoto": 19,
    "sasaki": 20,
    "shinchi": 21,
    "somu": 22,
    "takahashi": 23,
    "task": 24,
    "wada": 25,
    "yamada": 26,
    "yuta":27
  }

  #exts = ['.PNG','.JPG','.JPEG']
  exts = ['.JPG','.JPEG']

  for dirpath, dirnames, filenames in os.walk(outdir):
    for dirname in dirnames:
      if dirname in names:
        n = names[dirname]
        member_dir = os.path.join(dirpath, dirname)
        for dirpath2, dirnames2, filenames2 in os.walk(member_dir):
          if not dirpath2.endswith(dirname):
            continue
          for filename2 in filenames2:
            (fn,ext) = os.path.splitext(filename2)
            if ext.upper() in exts:
              img_path = os.path.join(dirpath2, filename2)
              print ('%s %s' % (img_path, n))