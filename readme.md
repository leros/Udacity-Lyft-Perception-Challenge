#[Lyft Perception Challenge](https://www.udacity.com/lyft-challenge)
June 2018

### Task

 Pixel-wise identification of objects in camera images. Specifically, identifying cars and the drivable area of the road.

### Analysis
- It's a semantic segmentation problem. [Here](https://github.com/tangzhenyu/SemanticSegmentation_DL) is a good summary of the cutting-edge research, papers, and codes on this topic.
- For this task, only three types of objects should be identified: vehicle, road, and all others, so it's a pixel-wise ternary classification problem.
- It's an extension of the Semantic Segmentation project.
- Inferencing speed is as important as the training accuracy/recall.

### Solution
- Backbone model: VGG16-based [FCN-8s](https://arxiv.org/pdf/1605.06211)
- Datasets: 1 official dataset and 2 additional datasets got from the slack channel.
- Loss: cross entropy loss and regularization loss
- Hyperparameters: 
  - learning rate: 1e-5
  - batch size: 1
  - epochs: 5
- Optimizer: Adam Optimizer
- Running environment: 
  - AWS p3.2xlarge instance running udacity-carnd-advanced-deep-learning AMI. 
  - Tensorflow 1.2.1
- Saved graph: stored with AWS S3 and available by running `preinstall_script.sh`

### Potential Improvements:
- Training: 
  - deal with unbalanced data, e.g. weighted lables
  - try different base model than VGG16
  - fine tune Hyperparameters
  - collect more and better data
  - validate and test model thoroughly 
- Inferring
  - freeze model properly (was not working as inferring with frozen model actually decrease FPS a lot)
  - optimize video/image processing