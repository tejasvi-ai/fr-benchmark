#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

from IPython import embed
from matplotlib import pyplot as plt
import openface
import numpy as np
import os
import itertools
import cv2
import argparse
import time
from collections import defaultdict
import json

start = time.time()


np.set_printoptions(precision=2)


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, "..", "models")
dlibModelDir = os.path.join(modelDir, "dlib")
openfaceModelDir = os.path.join(modelDir, "openface")

parser = argparse.ArgumentParser()

parser.add_argument("--imgs", type=str, nargs="+", help="Input images.", default=None)
parser.add_argument(
    "--dlibFacePredictor",
    type=str,
    help="Path to dlib's face predictor.",
    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"),
)
parser.add_argument(
    "--networkModel",
    type=str,
    help="Path to Torch network model.",
    default=os.path.join(openfaceModelDir, "nn4.small2.v1.t7"),
)
parser.add_argument("--dir", type=str, help="Nested directory", default=None)
parser.add_argument("--imgDim", type=int, help="Default image dimension.", default=96)
parser.add_argument("--verbose", action="store_true")

args = parser.parse_args()

if args.verbose:
    print(
        "Argument parsing and loading libraries took {} seconds.".format(
            time.time() - start
        )
    )

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print(
        "Loading the dlib and OpenFace models took {} seconds.".format(
            time.time() - start
        )
    )


def getRep(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        print("Face not found for", imgPath)
        raise Exception("Unable to find a face: {}".format(imgPath))
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))

    start = time.time()
    alignedFace = align.align(
        args.imgDim, rgbImg, bb, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE
    )
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    rep = net.forward(alignedFace)
    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        print("Representation:")
        print(rep)
        print("-----\n")
    return rep


if not args.imgs:
    from glob import glob

    f_list = glob(args.dir + "/*/*")
    d = defaultdict(list)
    for f in f_list:
        d[f.split("/")[-2]].append(f)
    args.imgs = f_list

    # trans_d = json.loads(args.json)
    # t_list = glob(args.tdir + "/*")


size = 200

tot_im = not_found = 0

fp_list = [v[0] for v in d.values()]
fp_dist = []
comb = list(itertools.combinations(fp_list, 2))
sz = size
for i, (img1, img2) in enumerate(comb):
    if i == sz:
        break
    print("\r", float(i) / len(comb) * 100, end="")
    try:
        tot_im += 1
        d_l1 = getRep(img1) - getRep(img2)
    except:
        not_found += 1
        sz += 1
        continue
    dist = np.dot(d_l1, d_l1)
    fp_dist.append((dist, (img1, img2)))


tp_dist = []
comb = [v[:2] for v in d.values()]
sz = size
for i, (img1, img2) in enumerate(comb):
    if i == sz:
        break
    print("\r", float(i) / len(comb) * 100, end="")
    try:
        tot_im += 1
        d_l1 = getRep(img1) - getRep(img2)
    except:
        not_found += 1
        sz += 1
        continue
    dist = np.dot(d_l1, d_l1)
    tp_dist.append((dist, (img1, img2)))

print("\nNot found", not_found, "images out of", tot_im)

false_positives = []
true_positives = []
for thresh in np.arange(0.3, 1.3, 0.05):
    false_positive = np.mean([dist[0] < thresh for dist in fp_dist]) * 100
    true_positive = np.mean([dist[0] < thresh for dist in tp_dist]) * 100

    false_positives.append(false_positive)
    true_positives.append(true_positive)

print(zip(false_positives, true_positives))

plt.plot(false_positives, true_positives, color="red", marker="o")
plt.savefig("plot2")

fp_dist.sort(key=lambda x: x[0])
tp_dist.sort(key=lambda x: x[0], reverse=True)

print("Worst false positives\n", fp_dist[:3])
print("Worst true positives\n", tp_dist[:3])


embed()
