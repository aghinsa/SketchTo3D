# OpenSFM Docker file python2

Runs Opensfm on python2.

## Step 1: Build docker image

From the root of the repo, run:
`docker build -t iitmcvg/opensfm opensfm`

## Step 2: Run container

Run

```
nvidia-docker run -it --rm --name $USERNAME \
-p 1002:22 \
-e USERNAME=$USERNAME \
-v /media/home/$USERNAME$:/home/$USERNAME \
-v /media/ssd/$USERNAME$:/mnt/ssd \
iitmcvg/opensfm bash
```
## Opensfm Instructions


Running

An example dataset is available at data/berlin.

1. Put some images in data/DATASET_NAME/images/

2. Put config.yaml in data/DATASET_NAME/config.yaml

3. Go to the root of the project and run bin/opensfm_run_all data/DATASET_NAME

4. Start an http server from the root with `python -m SimpleHTTPServer`.

5. Tunnel an appropriate port. Example:
```
ssh -R 18000:localhost:8000 -p PORT holojest@IP
```

Browse 
```
http://localhost:18000/viewer/reconstruction.html#file=/data/DATASET_NAME/reconstruction.meshed.json.
```