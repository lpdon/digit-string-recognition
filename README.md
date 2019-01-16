# Digit String Recognition

## Data

First download the CAR-A and CAR-B dataset from http://www.orand.cl/en/icfhr2014-hdsr/#datasets.  
Or use your own dataset and adapt its structure to match that of the CAR datasets.

## Train

The training can be started by running the `train.py` script. For example

```bash
python train.py --data '/path/to/ORAND-CAR-2014/CAR-A/' --epochs 20
```

Check `python train.py -h` for all possible options.

## Test Model

TODO

## Results

After training 100 epochs:

CAR-A
Test         | avg_dist:   0.108615 | accuracy:   0.905127 | avg_loss:   0.485466

CAR-B
Test         | avg_dist:   0.079289 | accuracy:   0.928230 | avg_loss:   0.315886