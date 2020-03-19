# CovidHelper

CovidHelper is a program inspired by Adrian Rosebrock tutorials and aims to become a reliable source of automated detection of CT and RX scans to find infections caused by COVID19. 

We can train a model, serialize and save to disk. We can then use load_module.py file to test if an image is positive or negative.

*TODO:*

1) Search and add Covid positive RX and CT scans to dataset for training
2) Search and add metadata in csv format
3) Search and add Normal RX and CT scans to dataset for training
4) Search and add Covid and Normal RX and CT scans for testing (must NOT be the same images included in training)
5) Refine the model and test reliability on field study cases.
