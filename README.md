Detecting vanishing points in images is useful for things like adjusting cameras, understanding scenes, or even for self-driving cars. Deep learning, a type of computer learning, is great at finding these vanishing points, but it has some problems:

1. It needs lots of labeled pictures, which is expensive and can have mistakes.
2. It requires powerful computers to process all the data.
3. When we change how we collect data, like taking pictures in different places or lighting conditions, the deep learning system might not work as well.
4. Even small changes in the problem, like looking for different numbers of vanishing points, mean we have to change the whole deep learning setup.

To solve these issues, we came up with a solution: we added some basic geometric rules to the deep learning process. Specifically, we used two ideas: the Hough Transform and the Gaussian sphere mapping.

- The Hough Transform helps us find lines in pictures.
- The Gaussian sphere mapping helps us find where lines meet

Training 
- `python train.py -d 0 --identifier baseline config/nyu.yaml`

Testing
-  `!python eval_nyu.py -d 0  --dump result/  config/nyu.yaml  logs/240410-192351-baseline/checkpoint_latest.pth.tar`
