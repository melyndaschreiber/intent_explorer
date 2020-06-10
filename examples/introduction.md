# Load the Datasets

```
from datasets import ulm_execution, ulm_imagination, sci_online, sci_offline, way_eeg_gal
```

There are four different datasets in this package that are easily downloadable, 
1) Upper Limb Movement Execution of Healthy Individuals
```
ulm_execution()
```
2) Upper Limb Movement Imagination of Healthy Individuals
```
ulm_imagination()
```
3) Upper Limb Movement Imaginations of Spinal Cord Injuries Offline
```
sci_offline()
```
4) Upper Limb Movement Imaginations of Spinal Cord Injuries Online
```
sci_online()
```
5) WAY-EEG-GAL 
```
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
