# Third-party notices

Pi0.5 is the model of Physical Intelligence; SMA is our training-time allocation
method. This repository does not claim authorship of Pi0.5 or OpenPI.

The upstream model and training utilities are obtained from
https://github.com/Physical-Intelligence/openpi at commit
`15a9616a00943ada6c20a0f158e3adb39df2ccac`, under Apache-2.0.
The loss-hook patch modifies OpenPI `src/openpi/models/pi0.py`; modified lines
return the native loss and an additional detached-use real7 score. Our training
adapter uses OpenPI's native train-state initialization and optimizer APIs.
The Apache license is included in `THIRD_PARTY_LICENSES/Apache-2.0.txt`.
Model weights, Gemma components and datasets have their own upstream terms.

Evaluation uses the Apache-2.0 AllenAI VLA Evaluation Harness at
`4aeb4369640e8019d46af9534ce9b957e486ad38`. It is installed separately, not vendored.
LIBERO, robosuite, MuJoCo and custom environment/data assets are separately
provided dependencies under their respective terms. No weights or raw robot
demonstrations are redistributed by this code release.
