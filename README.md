### FaceNet
```
python3 facenet.py --dir=nested_image_directory
```

### OpenFace
Copy the image data to openface directory. Then,
```
bash openface.sh
python openface.py --dir=nested_image_directory
```

### Dlib
* Create embeddings for faces using [dlib/create_dlib_embeddings.py](dlib/create_dlib_embeddings.py).
* Benchmark the prediction using [dlib/dlib_embeddings.py](dlib/dlib_embeddings.py).