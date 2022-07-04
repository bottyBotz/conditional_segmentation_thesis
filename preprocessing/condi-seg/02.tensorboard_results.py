from packaging import version

import tensorboard as tb
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats

# major_ver, minor_ver, _ = version.parse(tb.__version__).release
# assert major_ver >= 2 and minor_ver >= 3, \
#     "This notebook requires TensorBoard 2.3 or later."
# print("TensorBoard version: ", tb.__version__)

# experiment_id = "c1KCv3X3QvGwaXfgX1c4tg"
# experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
# df = experiment.get_scGet alars()
# df