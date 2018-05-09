import tensorflow as tf

def greedy(actions):
    return tf.argmax(actions)

def boltzmann(actions):
    return tf.multinomial(actions, 1)

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)  # fixed off by one bug
        discounted.append(r)
    return discounted[::-1]

def gae(rewards, values, bootstrap_value, gamma, dones):
    v = values + [bootstrap_value]
    adv = rewards + gamma * v[1:] - v[:-1]
    return discount_with_dones(rewards, dones, gamma)