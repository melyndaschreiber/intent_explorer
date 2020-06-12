# Load the Datasets

This module gives access to the datasets that are downloadable upon the following commands. Data in the form of zip files will be downloaded into a `tmp` folder in the current directory. Then the data files will be extracted into a `data` folder within the same directory. In addition to the data, the corresponding research paper and dataset descriptions are also available. After the data is extracted into the data folder, the `tmp` folder can be deleted.  

There are five different datasets in this package that are easily downloadable from their original sources. 

1) Upper Limb Movement Execution of Healthy Individuals
```
from datasets import ulm_execution
ulm_execution()
```
2) Upper Limb Movement Imagination of Healthy Individuals
```
from datasets import ulm_imagination()
ulm_imagination()
```
3) Upper Limb Movement Imaginations of Spinal Cord Injuries Offline
```
from datasets import sci_offline()
sci_offline()
```
4) Upper Limb Movement Imaginations of Spinal Cord Injuries Online
```
from datasets import sci_online()
sci_online()
```
5) WAY-EEG-GAL 
```
from datasets import way_eeg_gal()
way_eeg_gal()
```

# Load Individual Trials

```
import datasets
raw_execution = datasets.read_execution(subject = 1, trial = 1)
raw_imagination = datasets.read_imagination(subject = 1, trial = 1)
raw_sci_offline = datasets.sci_offline(subject = 1, trial = 1)
raw_sci_online = datasets.sci_online(subject = 1, trial = 1)
```
