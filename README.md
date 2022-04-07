# machine-learning-coursework2



should install libraries

pip install pandas
pip install numpy
pip install matplotlib
pip install scikit
pip install scikit-image
pip install IPython
pip install tensorflow



And use it like following:

```python
import ml_preprocessing

ml_preprocessing.model.fit(
    ml_preprocessing.train_generator,
    epochs = 30,
    steps_per_epoch =ml_preprocessing.nb_train_samples // ml_preprocessing.batch_size,
    validation_data =ml_preprocessing.valid_generator,
    validation_steps =ml_preprocessing.nb_valid_samples // ml_preprocessing.batch_size,
    verbose = 2,
    callbacks =ml_preprocessing.callbacks,
    shuffle = True
)
```