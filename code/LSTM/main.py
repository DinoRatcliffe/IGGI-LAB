from networks.lstm_network import LSTMNetwork
from networks.text_batcher import TextBatcher
import numpy as np
import tensorflow as tf

testing_string = "$A$B$C$D$E$F$G$H$I$J$K$L$M$N$O$P$Q$R$S$T$U$V$W$X$Y$Z$A$B$C$D$E$F$G$H$I$J$K$L$M$N$O$P$Q$R$S$T$U$V$W$X$Y$Z$A$B$C$D$E$F$G$H$I$J$K$L$M$N$O$P$Q$R$S$T$U$V$W$X$Y$Z"
epochs = 50 

training = True

config = {
            'model_dir': 'models/testing',
            'seq_length': 10,
            'batch_size': 10,
            'hidden_size': 200,
            'num_layers': 2,
            'keep_prob': 1.0,
            'gradient_clip': 10,
            'learning_rate': 1e-2,
            'training': training,
            'init_scale': 0.04
        }

if not training:
    config['batch_size'] = 1
    config['seq_length'] = 1

sequence_length = config['seq_length']
batch_size = config['batch_size']

tb = TextBatcher(testing_string, batch_size, sequence_length)
network = LSTMNetwork(config, len(tb.vocab))
network.start_session()

if training:
    for i in range(0, epochs):
        print("Running Epoch %d" % (i + 1))
        tb.reset()
        cost_sum = 0
        iter_count = 0
        while tb.has_next_batch():
            batch_inputs, batch_targets = tb.next_batch()
            cost, state = network.train(batch_inputs, batch_targets)

            cost_sum += cost
            iter_count += sequence_length

        print("Epoch: %d, Train Perplexity: %.3f" % (i + 1, np.exp(cost_sum/iter_count)))

    print("Saving model ...")
    network.save_model(i+1)
    print("Model saved")
else:
    prime_string = "$A$B$"
    prime_ids = tb.convert_string(prime_string)
    
    print tb.convert_ids(network.generate_text(prime_ids, 50))


network.end_session()
