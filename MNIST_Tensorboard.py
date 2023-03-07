
# coding: utf-8

# In[34]:


import os
import os.path
import tensorflow as tf 

LOGDIR="/tmp/mnsit_tutorial/"
LABELS=os.path.join(os.getcwd(),"labels_1024.tsv")
SPRITES=os.path.join(os.getcwd(),"sprite_1024.png")

mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + "data", one_hot=True)
### Get a sprite and labels file for the embedding projector ###


if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
    print("Necessary data files were not found. Run this command from inside the "
    "repo provided at "
    "https://github.com/dandelionmane/tf-dev-summit-tensorboard-tutorial.")


def conv_layer(input,size_in,size_out,name="conv"):

    with tf.name_scope(name):

        w=tf.Variable(tf.truncated_normal([5,5,size_in,size_out],stddev=0.1),name="W")

        b=tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")

        conv=tf.nn.conv2d(input,w,strides=[1,1,1,1],padding="SAME")

        act=tf.nn.relu(conv + b)

        tf.summary.histogram("weights",w)

        tf.summary.histogram("biases",b)

        tf.summary.histogram("activations",act)

        return tf.nn.max_pool(act,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")



def fc_layer(input,size_in,size_out,name="fc"):

    with tf.name_scope(name):

        w=tf.Variable(tf.truncated_normal([size_in,size_out],stddev=0.1),name="W")

        b=tf.Variable(tf.constant(0.1,shape=[size_out]),name="B")

        act=tf.matmul(input,w) + b 

        tf.summary.histogram("weights",w)

        tf.summary.histogram("biases",b)

        tf.summary.histogram("activations",act)

        return act

def mnist_model(learning_rate,use_two_fc,use_two_conv,hparam):

    tf.reset_default_graph()
    sess=tf.Session()

    # Setup placeholders, and reshape the data
    x=tf.placeholder(tf.float32,shape=[None,784],name="x")
    x_image=tf.reshape(x,[-1,28,28,1])

    tf.summary.image('input',x_image,3)
    y=tf.placeholder(tf.float32,shape=[None,10],name="labels")

    conv1=conv_layer(x_image,1,32,"conv1")
    conv_out=conv_layer(conv1,32,64,"conv2")

    flattened=tf.reshape(conv_out,[-1,7*7*64])

    fc1=fc_layer(flattened,7*7*64,1024,"fc1")

    relu=tf.nn.relu(fc1)

    tf.summary.histogram("fc1/relu",relu)

    logits=fc_layer(flattened,7*7*64,10,"fc")

    with tf.name_scope("xent"):

        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits,labels=y),name="xent")

        tf.summary.scalar("xent",xent)

    with tf.name_scope("train"):

        train_step=tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):

        correct_prediction=tf.equal(tf.argmax(logits,1),tf.argmax(y,1))

        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        tf.summary.scalar("accuracy",accuracy)

    summ=tf.summary.merge_all()

    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(LOGDIR + hparam)
    writer.add_graph(sess.graph)


    for i in range(2001):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            [train_accuracy, loss_val,s] = sess.run([accuracy,xent,summ], feed_dict={x: batch[0], y: batch[1]})
            tf.logging.debug('Step : {} | Train Accuracy : {} | Loss : {} '.format(i,train_accuracy,loss_val))
            writer.add_summary(s, i)
            saver.save(sess, os.path.join(LOGDIR, "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate,use_two_fc,use_two_conv):

    conv_param="conv=2" if use_two_conv else "conv=1"
    fc_param="fc=2" if use_two_fc else "fc=1"

    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():


    hparam=make_hparam_string(learning_rate=learning_rate,use_two_fc=True,use_two_conv=True)
    print('Starting run for %s' % hparam)


    mnist_model(learning_rate, use_two_fc=True, use_two_conv=True, hparam=hparam)

    print('Done training!')
    print('Run `tensorboard --logdir=%s` to see the results.' % LOGDIR)


# In[35]:


tf.logging.set_verbosity(tf.logging.DEBUG)
learning_rate=1E-3

if __name__=='__main__':
    main()


# In[36]:


tf.reset_default_graph()

