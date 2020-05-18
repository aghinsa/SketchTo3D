# Holojest

* ## [Introduction](#introduction)  
* ## [Training](#training)  
* ## [Testing](#testing) 
* ## [Fusion](#fusion)   
* ## [Citation](#citation)  
* ## [Further Improvements](#improvements)  

### <a name="introduction"></a> Introduction  
Uses deep learning to convert 2d pencil drawings to 3d model.Based on [this paper](#citation).We have coded the paper to predict models for characters only.The model takes input Front(F) and side(S)
views of a character and outputs,12 different views;depth and normal maps for each view.The outputed images can be fused with [this](https://github.com/happylun/SketchModeling/tree/master/Fusion).  

### <a name="training"></a> Training
* Training data is available [here](https://people.cs.umass.edu/~zlun/papers/SketchModeling/).    
* Clone the repo (*root*).  
* Install dependencies for [gradient checkpointing](https://github.com/openai/gradient-checkpointing)  
    * `pip install -r root/requirements.txt`  
* Modify `root/Sketch/module/config.py`  
    * change *main_dir*  
    * uncomment everything between *isTraining* and *Loss Tuning*  
* Put sketch,dnfs folders in main_dir.
* Run `export PYTHONPATH="$PYTHONPATH:/root/Sketch/"`  
* Run  `root/Sketch/train_main.py`  
    * Checkpoints will be saved in *home/holojest/Sketch/checkpoints*

### <a name="testing"></a> Testing  
* Move checkpoints to *root/Sketch/checkpoints*  
* model_dir configuration
    * Put drawings in *model_dir/images*  
    * Rename front view to **sketch-F-0.png** and side view to **sketch-S-0.png**
* Change `saver.restore(sess,train_dir+'/model.ckpt-36500')` to latest  
* Run `python root/Sketch/runner.py -i path_to_model_dir/`
* Output details
    * *Model_dir/output/images* depth Images,normal maps,masks.
    *  *Model_dir/output/result* fused  

### <a name="fusion"></a> Fusion  
Follow [this](https://github.com/happylun/SketchModeling/tree/master/Fusion)  
> ReconstructMesh.exe 1 FS sketch_folder map_folder output_result_folder view.off  
  PoissonRecon.exe --in points.ply --out mesh.ply --depth 11 --samplesPerNode 5.0 --pointWeight 0.1    
   ReconstructMesh.exe 2 FS sketch_folder map_folder output_result_folder view.off  

* sketch_folder: 2 sketches  
* map_folder : maps,masks,depths  
* output_folder : anything  


### <a name="citation"></a> Citation  
> Zhaoliang Lun, Matheus Gadelha, Evangelos Kalogerakis, Subhransu Maji, Rui Wang,  
"3D Shape Reconstruction from Sketches via Multi-view Convolutional Networks",   
Proceedings of the International Conference on 3D Vision (3DV) 2017  

* https://github.com/happylun/SketchModeling
* https://people.cs.umass.edu/~zlun/SketchModeling/
* https://arxiv.org/pdf/1707.06375.pdf  

### <a name="improvements"></a> Further Improvments  
* Improve adverserial loss.
* Find a better GAN implementation.  
* Output 14 views,or psued it.  


