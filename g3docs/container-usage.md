# Container Usage

## SSH Access

`ssh -R 52698:localhost:52698 -p PORT holojest@IP`

Replace `PORT` and `IP` by the respective ones.

## Moving and Copying files

`scp -R 52698:localhost:52698 -p PORT holojest@IP`

## Imagecat

Cat for Images.

```
sudo curl -o /usr/local/bin/imgcat -O https://raw.githubusercontent.com/gnachman/iTerm2/master/tests/imgcat && sudo chmod +x /usr/local/bin/imgcat
```

## Running Opensfm

Running

An example dataset is available at data/berlin.

1. Put some images in data/DATASET_NAME/images/

2. Put config.yaml in data/DATASET_NAME/config.yaml

3. Go to the root of the project and run bin/opensfm_run_all data/DATASET_NAME

4. Start an http server from the root with `python -m SimpleHTTPServer`.

Browse 
```
http://localhost:8000/viewer/reconstruction.html#file=/data/DATASET_NAME/reconstruction.meshed.json.
```

5. Tunnel an appropriate port. Example:
```
ssh -R 18000:localhost:8000 -p PORT holojest@IP
```

Browse 
```
http://localhost:18000/viewer/reconstruction.html#file=/data/DATASET_NAME/reconstruction.meshed.json.
```

Things you can do from there:

* Use datasets with more images
* Click twice on an image to see it. Then use arrows to move between images.

## Rmate

See (dockerfiles/rmate.md) on the documentation engine.

## Common Issues

* Pip installs for `pip >=10.0.1`  
     Run as `python -m pip ...`  
     Or `python3 -m pip ...`
