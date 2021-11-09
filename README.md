# Online-compatible Unsupervised Non-resonant Anomaly Detection Repository

Repository containing all scripts used in the studies of Online-compatible Unsupervised Non-resonant Anomaly Detection model.

To train your own model, first Download the official dataset from [zenodo](https://zenodo.org/record/5046389#.YNyFtRMzZqt) and use the [example code](https://github.com/mpp-hep/ADC2021-examplecode) to prepare the datasets. To run the training, use:

```bash
python AE40Mhz.py [--single/--double/--supervised/--all] [--load] --out NAME
```

To train a single AE, the double + decorrelatied method, supervisedd, or all of them respectively. Trained model weights are also providedd in the ```weights``` folder that can be loaded using the ```--load``` flag. 

The output of the script will create an ```NAME.h5``` file in the base directory. Use this file to plot the results using the script ```plot.py```

```bash
python plot.py --file NAME.h5
```

Different plot options are available in the script.