# Concurrent-Video-Processing-Pipeline

This repository illustrates the use of a multithreaded video processing pipeline. 
There are several advantages of concurrent execution of a video processing pipeline over the sequential one
Modern day video processing pipelines may make use of deep learning based computer vision models such as CNN classifiers, Object Detection networks, etc
One forward propagation of these networks can massively slow down the entire video processing pipeline due to the large number of computations involved in one forward pass. To address this issue we make use of multithreading to put IO bound process and the CPU bound process into separate threads

![image](https://user-images.githubusercontent.com/38568261/170887780-60f4de80-5b58-4729-9e4a-8e335b9853bf.png)

1. The Multithreaded implementation of a video processing pipeline has a much better FPS than the sequential one
2. Effectively utilizes the CPU resources
3. Does not let the IO bound instructions (reading video frames/displaying video frames) come in the way of processing
