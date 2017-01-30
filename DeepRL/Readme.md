# Asynchronous Actor Critic (A3C) Thensorflow implementation

## Dependencies
- Python 2.x
- TensorFlow r0.12
- Numpy

## How to run
To run the algorithm using the default parameters

```
$ python run_agent.py
```

- The final model metadata will be saved at: 
*./model/\<openai_gym_game_name\>/*

- Tensorboard information regarding **Average score per episode** behaviour during training as well as **losses** and **learning rate dacay** curves will be saved at: *./summary/\<openai_game_name\>/train/*
