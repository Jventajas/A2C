import tensorflow as tf


def greedy(actions):
    return tf.argmax(actions)


def boltzmann(actions):
    return tf.multinomial(actions, 1)


class ActorCritic:

    def __init__(self, n_actions, selection_strategy=boltzmann):
        conv1 = tf.layers.Conv2D(32, 5, activation=tf.nn.relu)
        pool1 = tf.layers.MaxPooling2D(2, 2)
        conv2 = tf.layers.Conv2D(64, 5, activation=tf.nn.relu)
        pool2 = tf.layers.MaxPooling2D(2, 2)
        flatten = tf.layers.Flatten()
        fc_policy = tf.layers.Dense(n_actions, activation=None)
        fc_value = tf.layers.Dense(1, activation=None)

        def forward(obs):
            with tf.device('/cpu:0'):
                x = tf.constant(obs, dtype=tf.float32)
                x = conv1(x)
                x = pool1(x)
                x = conv2(x)
                x = pool2(x)
                return flatten(x)

        def action(obs):
            with tf.device('/cpu:0'):
                x = forward(obs)
                x = fc_policy(x)
                x = tf.nn.softmax(x)
                x = selection_strategy(x)
                return x.numpy()

        def value(obs):
            with tf.device('/cpu:0'):
                x = forward(obs)
                x = fc_value(x)
                return x.numpy()

        def step(obs):
            with tf.device('/cpu:0'):
                x = forward(obs)
                policy = fc_policy(x)
                policy = tf.nn.softmax(policy)
                act = selection_strategy(policy)
                val = fc_value(x)
                return act.numpy(), val.numpy()

        def learn(obs, rewards, masks, actions, values):
            with tf.device('/cpu:0'):
                advs = rewards - values
                logits = forward(obs)
                probs = tf.nn.softmax(logits)
                xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
                policy_loss = tf.reduce_mean(xentropy * advs)
                value_loss = tf.reduce_mean(tf.square(advs))
                entropy_loss = tf.reduce_mean(probs * tf.log(probs))
                total_loss = policy_loss + value_loss - entropy_loss



        self.action = action
        self.value = value
        self.step = step
        self.learn = learn
