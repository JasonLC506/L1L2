# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# import os
# os.makedirs("./ckpt_test")

# <codecell>

import tensorflow as tf
import numpy as np
import math
from datetime import datetime
from datetime import timedelta

## ======== file path ============= ##
checkpointdir = "./ckpt_L1_test/"
checkpointfiles = {"whole": "whole.ckpt", "embed": "embed.ckpt", "hidden1": "hidden1.ckpt", "hidden2": "hidden2.ckpt"}
for key in checkpointfiles.keys():
    checkpointfiles[key] = checkpointdir + checkpointfiles[key]
	
datadir = "../../wilsonlym/IST597/"
L1_train = datadir + "english_sent_train.csv"
L1_valid = datadir + "english_sent_valid.csv"
L1_test = datadir + "english_sent_test.csv"
L2_train = datadir + "spanish_sent_train.csv"
L2_valid = datadir + "spanish_sent_valid.csv"
L2_test = datadir + "spanish_sent_test.csv"

resultdir = "./result/"
resultfile_L1_name = resultdir + "result_L1.txt"
## ================================ ##

## ====== hyperperameter ===== ##
batch_size = 128
embedding_size = 128
hidden_size1 = 64
hidden_size2 = 128
window_size = 5
voca_size_L1 = 26651
voca_size_L2 = 40405
Learning_rate = 0.001
epoch_max = 1000
batch_num = 100000000
## ============================ ##

def loss_xentropy(logits, labels):
    """Calculates the loss from the logits and the labels.
      Args:
        logits: Logits tensor, float - [batch_size, NUM_CLASSES].
        labels: Labels tensor, int32 - [batch_size].
      Returns:
        loss: Loss tensor of type float.
      """
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits, labels, name = "xentropy")
    loss = tf.reduce_mean(cross_entropy, name = "xentropy_mean")
    return loss

def traintest_L1(batch_size = 128, embedding_size = 128, hidden_size1 = 128, hidden_size2 = 128, window_size = 5, voca_size = 26651, 
              Learning_rate = 0.001, epoch_max = 10, batch_num = 100,
              datafile_train = "english_sent_train.csv", datafile_valid = "english_sent_valid.csv"):
    
    graph = tf.Graph()
    with graph.as_default():
        
        ## input placeholder ##
        train_inputs = tf.placeholder(tf.int32, shape = [batch_size, window_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size]) # label is represented by single int
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
        ## inner layers ##
        # embedding layer #
        embeddings = tf.Variable(tf.random_uniform([voca_size, embedding_size], -1.0, 1.0), name = "embeddings")
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        flat_embedding_size = window_size*embedding_size
        embed_flat = tf.reshape(embed, [batch_size, flat_embedding_size]) 
        # hidden layer1 #
        weights_h1 = tf.Variable(
                                tf.truncated_normal([flat_embedding_size, hidden_size1], 
                                                    stddev = 1.0/math.sqrt(flat_embedding_size)), name = "weights_h1")
        biases_h1 = tf.Variable(tf.zeros([hidden_size1]), name = "biases_h1")  
        hidden1 = tf.nn.relu(tf.matmul(embed_flat, weights_h1) + biases_h1)
        
        # hidden layer2 #
        weights_h2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2],
                                                     stddev = 1.0/math.sqrt(hidden_size1)), name = "weights_h2")
        biases_h2 = tf.Variable(tf.zeros([hidden_size2]), name = "biases_h2")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_h2) + biases_h2)
        ## output layer ##
        weights_o = tf.Variable(
                                tf.truncated_normal([hidden_size2, voca_size], 
                                                    stddev = 1.0/math.sqrt(hidden_size2)))
        biases_o = tf.Variable(tf.zeros([voca_size]))
        logits = tf.matmul(hidden2, weights_o) + biases_o
        
        ## optimizer ##
        loss = loss_xentropy(logits, train_labels)
        optimizer = tf.train.RMSPropOptimizer(Learning_rate).minimize(loss)
        
        ## initializer ##
        init = tf.initialize_all_variables()
        
        ## saver ##
        # saver_whole for interupt recover#
        saver_whole = tf.train.Saver(name = "whole", max_to_keep = 3,
                                     keep_checkpoint_every_n_hours = 12.0)
        # saver_embedding#
        saver_embed = tf.train.Saver(name = "embed", max_to_keep = 3,
                                     var_list = [embeddings])
        # saver_hidden1 for reuse#
        saver_hidden1 = tf.train.Saver(name = "hidden1", max_to_keep = 20,
                                     var_list = [weights_h1, biases_h1])
        #saver_hidden2 for reuse#
        saver_hidden2 = tf.train.Saver(name = "hidden2", max_to_keep = 20,
                                     var_list = [weights_h2, biases_h2])
        
    with tf.Session(graph = graph) as session:
        
        init.run()
        print("initialized")
        
#         ### restore test ###
#         print("biases_h1:", biases_h1.eval())
        
#         ### restore test ##
#         saver_hidden1.restore(session, "hidden1.ckpt-0")
#         print("weights_h1:", weights_h1.eval()[1,:])
#         print("biases_h1:", biases_h1.eval())

        for epoch in xrange(epoch_max):
            average_loss = 0
            step = 0
            # get training batch #
            batches = data_generator(datafile_train, window=window_size, batch_size=batch_size, batch_num=batch_num)
            
            # timer #
            start = datetime.now()
            
            for batch_inputs, batch_labels in batches:
                
#                 ### test
#                 print batch_inputs.shape
#                 print batch_labels.shape
                

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels} 
                
                ## run ##
                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
                average_loss += loss_val
                
                step +=1
                if step % 100 == 0:
                    print("loss at step ", step, " : ", loss_val)
                    print("average loss until now: ", average_loss/step)
            
            duration = datetime.now() - start
            print ("time for epoch", epoch,":",duration.total_seconds())
            
            # save for each epoch #
            saver_whole.save(session, checkpointfiles["whole"], global_step = epoch)
            saver_embed.save(session,checkpointfiles["embed"], global_step = epoch)
            saver_hidden1.save(session, checkpointfiles["hidden1"], global_step = epoch)
            saver_hidden2.save(session, checkpointfiles["hidden2"], global_step = epoch)
            
            # validate (test ?) #
            valid = data_generator(datafile_valid, window=window_size, batch_size=batch_size, batch_num=batch_num*10)
            loss_valid_avg = 0
            iters = 0
            for valid_inputs, valid_labels in valid:
                loss_valid = session.run(loss, feed_dict = {train_inputs: valid_inputs, train_labels: valid_labels})
                loss_valid_avg += (loss_valid*batch_size)
                iters += 1
            print("valid log-perplexity in epoch ", epoch, ": ", loss_valid_avg/(iters*batch_size))
			
            resultfile = open(resultfile_L1_name,"a")
            resultfile.write("valid log-perplexity in epoch %d: %f12.6" % (epoch,loss_valid_avg/(iters*batch_size)))
            resultfile.close()
                
        ### restore test ###        
#         print ("after training, weights_h1:", weights_h1.eval()[1,:])
#         print ("after training, biases_h1", biases_h1.eval())
            

# <codecell>

def data_generator(filename, window=5, voca_size=26651, batch_size=32, batch_num=20):
    """
    without random shuffle now
    only first 1000000 for training
    """
    X = []
    Y = []
    last_sent = []
    cnt = 0
    with open(filename, 'r') as f:
        for line in f:
            arr = line.strip().split('\t')
            arr = last_sent + arr
            arr = map(int, arr)
            arr_len = len(arr)
            #X += [arr[i:i+window] for i in xrange(arr_len - window)]
            for i in xrange(window, arr_len):
                X.append(arr[i - window: i])
#                 Y_ = [0] * voca_size
#                 Y_[arr[i]] = 1
                Y.append(arr[i])
                cnt += 1
                if cnt % batch_size == 0:
                    X = np.array(X)
                    Y = np.array(Y)
                    batch_num += -1
                    yield X, Y
                    X = []
                    Y = []
                if batch_num <= 0:
                    return
            last_sent = arr[-window:]
            ### only first 1000000
            #if cnt >= 1000000:
            #    break

# <codecell>


# <codecell>

def traintest_L2(batch_size = 128, embedding_size = 128, hidden_size1 = 64, hidden_size2 = 128, window_size = 5, voca_size = 40403, 
              Learning_rate = 0.01, epoch_max = 10, batch_num = 100, hidden_add_size1 = 32, hidden_add_size2 = 64, 
              epoch_L1 = 0, datafile_train = "spanish_sent_train.csv", datafile_valid = "spanish_sent_valid.csv"):
    
    graph = tf.Graph()
    with graph.as_default():
        
        ## ========= build the graph ========== ##
        ## input placeholder ##
        train_inputs = tf.placeholder(tf.int32, shape = [batch_size, window_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size]) # label is represented by single int
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
        ## inner layers ##
        # embedding layer #
        embeddings = tf.Variable(tf.random_uniform([voca_size, embedding_size], -1.0, 1.0), name = "embeddings")
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        flat_embedding_size = window_size*embedding_size
        embed_flat = tf.reshape(embed, [batch_size, flat_embedding_size]) 
        
        # hidden layer1 #
        # --- from L1 trainable (?)--- #
        weights_h1 = tf.Variable(
                                tf.truncated_normal([flat_embedding_size, hidden_size1], 
                                                    stddev = 1.0/math.sqrt(flat_embedding_size)), name = "weights_h1",
                                trainable = False)
        biases_h1 = tf.Variable(tf.zeros([hidden_size1]), name = "biases_h1", trainable = False)  
        hidden1 = tf.nn.relu(tf.matmul(embed_flat, weights_h1) + biases_h1)
        # --- block --- #
        weights_bh1 = tf.Variable(
                                tf.truncated_normal([flat_embedding_size, hidden_add_size1], 
                                                    stddev = 1.0/math.sqrt(flat_embedding_size)), name = "weights_bh1")
        biases_bh1 = tf.Variable(tf.zeros([hidden_add_size1]), name = "biases_bh1")  
        hidden_b1 = tf.nn.relu(tf.matmul(embed_flat, weights_bh1) + biases_bh1)
        
        # hidden layer2 #
        # --- from L1 untrainable --- #
        weights_h2 = tf.Variable(tf.truncated_normal([hidden_size1, hidden_size2],
                                                     stddev = 1.0/math.sqrt(hidden_size1)), name = "weights_h2",
                                 trainable = False)
        biases_h2 = tf.Variable(tf.zeros([hidden_size2]), name = "biases_h2", trainable = False)
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights_h2) + biases_h2)
        # --- block --- #
        weights_bh2_b2b = tf.Variable(
                                      tf.truncated_normal([hidden_add_size1, hidden_add_size2],
                                                          stddev = 1.0/math.sqrt(hidden_add_size1)), name = "weights_bh2_b2b")
        weights_bh2_h2b = tf.Variable(
                                      tf.truncated_normal([hidden_size1, hidden_add_size2],
                                                          stddev = 1.0/math.sqrt(hidden_size1)), name = "weights_bh2_h2b")
        biases_bh2 = tf.Variable(tf.zeros([hidden_add_size2]), name = "biases_bh2")
        hidden_b2 = tf.nn.relu(tf.matmul(hidden1, weights_bh2_h2b) + tf.matmul(hidden_b1, weights_bh2_b2b) + biases_bh2)
        
        # output layer #
        weights_o_h2o = tf.Variable(
                                    tf.truncated_normal([hidden_size2, voca_size],
                                                        stddev = 1.0/math.sqrt(hidden_size2)), name = "weights_o_h2o")
        weights_o_b2o = tf.Variable(
                                    tf.truncated_normal([hidden_add_size2, voca_size],
                                                        stddev = 1.0/math.sqrt(hidden_add_size2)), name = "weights_o_b2o")
        biases_o = tf.Variable(tf.zeros([voca_size]))
        logits = tf.matmul(hidden2, weights_o_h2o) + tf.matmul(hidden_b2, weights_o_b2o) + biases_o
        ## ========================================================================== ##
        
        ## optimizer ##
        loss = loss_xentropy(logits, train_labels)
        optimizer = tf.train.RMSPropOptimizer(Learning_rate).minimize(loss)
        
        ## initializer ##
        init = tf.initialize_all_variables()
        
        saver_whole = tf.train.Saver(name = "whole", max_to_keep = 3,
                                     keep_checkpoint_every_n_hours = 12.0)
        
        # saver_embedding#
        saver_embed = tf.train.Saver(name = "embed", max_to_keep = 3,
                                     var_list = [embeddings])
        # saver_hidden1 for reuse#
        saver_hidden1 = tf.train.Saver(name = "hidden1", max_to_keep = 20,
                                     var_list = [weights_h1, biases_h1])
        #saver_hidden2 for reuse#
        saver_hidden2 = tf.train.Saver(name = "hidden2", max_to_keep = 20,
                                     var_list = [weights_h2, biases_h2])
        
    with tf.Session(graph = graph) as session:
        
        init.run()
        ## restore ##
        saver_hidden1.restore(session, "hidden1.ckpt-%d" % epoch_L1)
        saver_hidden2.restore(session, "hidden2.ckpt-%d" % epoch_L1)       
        
        for epoch in xrange(epoch_max):
            average_loss = 0
            step = 0
            # get training batch #
            batches = data_generator(datafile_train, window=window_size, batch_size=batch_size, batch_num=batch_num)
        
            for batch_inputs, batch_labels in batches:
                
                # timer #
                start = datetime.now()
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels} 
                
                ## run ##
                _, loss_val = session.run([optimizer, loss], feed_dict = feed_dict)
                average_loss += loss_val
                
                duration = datetime.now() - start
                
                step +=1
                if step % 100 == 0:
                    print("time for this step: ", duration.total_seconds())
                    print("loss at step ", step, " : ", loss_val)
                    print("average loss until now: ", average_loss/step)
            
            # for testing, by now #
            # validate (test ?) #
            valid = data_generator(datafile_valid, window=window_size, batch_size=batch_size, batch_num=batch_num)
            loss_valid_avg = 0
            iters = 0
            for valid_inputs, valid_labels in valid:
                loss_valid = session.run(loss, feed_dict = {train_inputs: valid_inputs, train_labels: valid_labels})
                loss_valid_avg += (loss_valid*batch_size)
                iters += 1
            print("valid log-perplexity in epoch ", epoch, ": ", loss_valid_avg/(iters*batch_size))

        

if __name__ == "__main__":
	
    traintest_L1(batch_size = batch_size, embedding_size = embedding_size, hidden_size1 = hidden_size1, hidden_size2 = hidden_size2, window_size = window_size, 
            Learning_rate = Learning_rate, epoch_max = epoch_max, batch_num = batch_num, 
            voca_size = voca_size_L1, datafile_train = L1_train, datafile_valid = L1_valid)

