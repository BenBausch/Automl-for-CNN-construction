# Automl-for-CNN-construction

## What is this repo about

This the final project of the automated machine learning class.
The goal was find good CNN configurations in an automated fashion for a downsampled verison of the flower dataset http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html
Two baselines were given: orange baseline with hypervolume of  and the red baseline with a hypervolume of 

## My Approach

I have choosen an evolutionary algorithm with 2 kinds of parent selection and multiple different recombination and mutation strategies to produce offspring.
The first sampling algorithm did sample a tournemant group from a distibution proportional to the inverse size (smallest network more likely) and 
then selected the best performing model as parent. The second parent selection algorithm did sample parents proportional to the hypervolume of the individual point w.r.t the same 
reference point, thereby optimizing both cost functions at the same time. Training was done using early stopping with a patience of 10, so good configurations were very unlikely to stop too early.

## Results

My algorithm found configurations strong then both baselines. Only for very small networks the performances of the baselines were better.
In the following, you can see the performance of my algorithm over a time span of 31 episodes (episode 9 is actually episode 10, this was due to a out of memory error):
 ![alt-text](https://github.com/BenBausch/Automl-for-CNN-construction/blob/master/src/pareto_fronts/pareto_fronts.gif)
