from xmps.fMPS import fMPS
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def to_mps(vec, D=None):
    vec = vec.reshape(-1)
    L = math.ceil(math.log(len(vec), 2))
    padded_vec = np.pad(vec, (0, 2**L-784)).reshape(*[2]*L)
    return fMPS().left_from_state(padded_vec).left_canonicalise(D)

k = 100
Ds = [32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2]
all_overlaps = []
for img in x_train[:k]:
    overlaps = [to_mps(img).left_canonicalise(D).overlap(to_mps(img))**2 for D in Ds]
    plt.plot(Ds, overlaps, c='lightgray', linewidth=0.5)
    all_overlaps.append(overlaps)

plt.plot(Ds, np.mean(np.array(all_overlaps), 0), color='red')
plt.ylabel('$|\\langle \\mathrm{img}(D) | \\mathrm{img}\\rangle|^2$')
plt.xlabel('$D$')
plt.savefig('overlaps.pdf')

