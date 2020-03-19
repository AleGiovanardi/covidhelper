# CovidHelper

CovidHelper is a program inspired by Adrian Rosebrock tutorials and aims to become a reliable source of automated detection of CT and RX scans to find infections caused by COVID19. 

We can train a model, serialize and save to disk. We can then use load_module.py file to test if an image is positive or negative.

*TODO:*

1) Search and add Covid positive RX and CT scans to dataset for training
2) Search and add metadata in csv format
3) Search and add Normal RX and CT scans to dataset for training
4) Search and add Covid and Normal RX and CT scans for testing (must NOT be the same images included in training)
5) Refine the model and test reliability on field study cases.
6) Add Grad-CAM debugging module to test if model is actully looking for what we are looking for!


## Getting Started

Build a dataset with build_covid_dataset.py
Train your model with train_covid19.py
Test your model against candidates with load_model.py
Debug your model with Grad-CAM debug using apply_gradcam.py

### Prerequisites

- Python3

- OpenCV

- Tensorflow

- Keras

[...]install instructions coming soon[...]

### Installing

[...]install instructions coming soon[...]


## Running the tests

[...]coming soon[...]


## Contributing

Please read [CONTRIBUTING.md]() for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

* **Adrian Rosebrock** - *Initial work* - [PyImageSearch](https://github.com/jrosebr1)

* **Alessandro Giovanardi** - *This repo* - [](https://github.com/AleGiovanardi)

* **Joseph Paul Cohen** - *Initial DataSet Curator* - [PyImageSearch](https://github.com/ieee8023)

See also the list of [contributors](https://github.com/AleGiovanardi/covidhelper/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Wicked Hat tip to anyone whose code was used
* My mother which is an ER doctor fighting on the frontline against COVID19 plague in Italy
* My family and my friends


