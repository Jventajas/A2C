import tensorflow as tf
import tensorflow.contrib.eager as tfe
from utils import boltzmann

layers = tf.keras.layers


class ActorCritic(tf.keras.Model):

    def __init__(self, n_actions, selection_strategy=boltzmann, lr=0.001):
        super(ActorCritic, self).__init__()
        self.conv1 = layers.Conv2D(64, 5, padding='SAME', activation=tf.nn.relu)
        self.pool1 = layers.MaxPool2D(4, 4)
        self.conv2 = layers.Conv2D(128, 5, activation=tf.nn.relu)
        self.pool2 = layers.MaxPool2D(4, 4)
        self.flatten = layers.Flatten()
        self.policy = layers.Dense(n_actions, activation=None)
        self.value = layers.Dense(1, activation=None)

        self.selection_strategy = selection_strategy
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)


    def forward(self, obs):
        x = tf.constant(obs, dtype=tf.float32)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        policy = self.policy(x)
        value = self.value(x)
        return policy, value

    def call(self, obs):
        policy, value = self.forward(obs)
        policy = tf.nn.softmax(policy)
        action = self.selection_strategy(policy)
        return action.numpy(), value.numpy()

    def learn(self, obs, rewards, masks, actions, values):
        with tfe.GradientTape(persistent=True) as tape:
            advs = rewards - values
            policy, value = self.forward(obs)
            smpolicy = tf.nn.softmax(policy)
            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy, labels=actions)
            policy_loss = tf.reduce_mean(xentropy * advs)
            value_loss = tf.reduce_mean(tf.square(advs))
            entropy_loss = tf.reduce_mean(smpolicy * tf.log(smpolicy))
            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        grads = tape.gradient(total_loss, self.variables)
        grads, gl_norm = tf.clip_by_global_norm(grads, 5.0)
        weight_norm = tf.global_norm(self.variables)
        self.optimizer.apply_gradients(zip(grads, self.variables))
