from torch import cuda
from lm_trainer import LMTrainer
print("main script:")
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(cuda.is_available())
print(cuda.device_count())

