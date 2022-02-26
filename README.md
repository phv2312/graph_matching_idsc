# Graph Matching Line-Art Colorization

## Features

- Graph matching approach for line-art colorization
- Many-to-one constraint, greedy algorithm instead of hungarian in the original paper.
- We use IDSC instead of deep feature embedding.

## Entry point

```commandline
python main.py 
```

## Some results

We compare against the result of using node only & node + edge (spectral matching).

| reference image                                                | target image                                                 | node matching                        | spectral matching                        |
|----------------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| <img src="./data/suneo_image/reference_color.png" width="300"> | <img src="./data/suneo_image/target_sketch.png" width="300"> | <img src="./result_of_node_matching.png" width="300"> | <img src="./result_of_spectral_matching.png" width="300"> |

## References

- [Anime Sketch Colorization by Component-based Matching using Deep Appearance Features and Graph Representation](https://ieeexplore.ieee.org/abstract/document/9412507)
- [Shape Classification Using the Inner-Distance](https://www.cs.umd.edu/~djacobs/pubs_files/ID-pami-8.pdf)
