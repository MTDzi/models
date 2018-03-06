import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
import memory


KEY_DIM = 128
MEM_SZ = 2**10
VOCAB_SZ = 30


memory = memory.Memory(KEY_DIM, MEM_SZ, VOCAB_SZ)

batch = tf.constant(
    [KEY_DIM*[1], KEY_DIM*[1]],
    shape=[2, KEY_DIM],
    dtype=tf.float32
)

y = tf.constant([1, 0], dtype=tf.int32)

result, mask, teacher_loss = memory.query(query_vec=batch, intended_output=None)#y)
