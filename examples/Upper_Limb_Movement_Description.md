# Upper Limb Movement Dataset

## Download the Dataset and Information 

This module gives access to the datasets that are downloadable upon the following commands. Data in the form of zip files will be downloaded into a `tmp` folder in the current directory. Then the data files will be extracted into a `data` folder within the same directory. In addition to the data, the corresponding [research paper](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable) and [dataset description](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0182578&type=printable) are also available. After the data is extracted into the data folder, the `tmp` folder can be deleted manually. 

```
from datasets import ulm_execution
ulm_execution()
```

## Trial Information

* **subjects**: 15
* **trials**: 10
* **original events**: flexion, extension, open hand, close hand, pronation, supination

![montage_image](/images/ulm_montage.png)

## Load Individual Trials

```
import datasets
raw_execution = datasets.read_execution(subject = 1, trial = 1)
raw_imagination = datasets.read_imagination(subject = 1, trial = 1)
```

