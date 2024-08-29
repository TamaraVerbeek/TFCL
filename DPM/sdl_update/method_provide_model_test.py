import tensorflow as tf
import numpy as np
import time
import torch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

class Task_free_continual_learning_provide_model:
    def __init__(self,
                verbose=False,
                seed=123,
                dev='cpu',
                dim=4,
                hidden_units=100,
                learning_rate=0.005,
                ntasks=1,
                gradient_steps=None,
                loss_window_length=5,
                loss_window_mean_threshold=0.2,
                loss_window_variance_threshold=0.1, 
                MAS_weight=0.5,
                recent_buffer_size=50,
                hard_buffer_size=50,
                model=None):
        # Define your model architecture here
        self.model = model
     
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # Initialize an Adam optimizer
        self.optimizer = Adam()

        # Initialize a categorical cross-entropy loss function
        self.loss_fn = CategoricalCrossentropy()


        self.verbose = True
        self.verbose=verbose
        self.dim=dim
        self.ntasks=ntasks
        self.gradient_steps=gradient_steps
        self.loss_window_length=loss_window_length
        self.loss_window_mean_threshold=loss_window_mean_threshold
        self.loss_window_variance_threshold=loss_window_variance_threshold
        self.MAS_weight=MAS_weight
        self.recent_buffer_size=recent_buffer_size
        self.hard_buffer_size=hard_buffer_size

    def method(self, data, use_hard_buffer=False, continual_learning=False):
        count_updates = 0
        stime = time.time()
        losses = []
        test_loss = {i: [] for i in range(self.ntasks)}
        future_losses = []
        recent_buffer = []
        hard_buffer = []
        loss_window = []
        loss_window_means = []
        loss_window_variances = []
        update_tags = []
        new_peak_detected = True
        star_variables = []
        omegas = []
        update_times=0


        for t in range(self.ntasks):
            for s in range(len(data.inputs[t])):
                recent_buffer.append({'state': data.inputs[t][s], 'trgt': data.labels[t][s]})
                if len(recent_buffer) > self.recent_buffer_size:
                    del recent_buffer[0]

                if len(recent_buffer) == self.recent_buffer_size:
                    msg = 'task: {0} step: {1}'.format(t, s)
                    print('recent buffer is full',s)
                    #return recent_buffer
                    x = np.asarray([_['state'] for _ in recent_buffer])
                    y = np.asarray([_['trgt'] for _ in recent_buffer])
                    #return x,y
                    #test on future data
                    xf=x[:]
                    yf=y[:]
                    yf_pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xf.T.tolist()])
                    accuracy = np.mean(np.argmax(yf_pred, axis=1) == yf)
                    future_losses.append(accuracy)

                    if use_hard_buffer and len(hard_buffer) != 0:
                        print('hard buffer is not empty',s)
                        xh = np.asarray([_['state'] for _ in hard_buffer])
                        yh = np.asarray([_['trgt'] for _ in hard_buffer])

                    for gs in range(self.gradient_steps):
                        with tf.GradientTape() as tape:
                            y_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in x.T.tolist()],training=True)
                            y_sup = tf.one_hot(tf.convert_to_tensor(y, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                            #print(y_sup)
                            recent_loss = self.loss_fn(y_sup, y_pred)
                            #recent_loss = torch.nn.MSELoss(reduction='none')(y_sup, y_pred)
                            #print(recent_loss)
                            total_loss = tf.reduce_sum(recent_loss)
                            if use_hard_buffer and len(hard_buffer) != 0:
                                #return xh
                                #yh_pred = self.model(tf.convert_to_tensor(xh, dtype=tf.float32))
                                yh_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xh.T],training=True)
                                yh_sup = tf.one_hot(tf.convert_to_tensor(yh, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                                hard_loss = self.loss_fn(yh_sup, yh_pred)
                                total_loss += tf.reduce_sum(hard_loss)

                            if gs == 0:
                                first_train_loss = total_loss.numpy()

                            if continual_learning and len(star_variables) != 0 and len(omegas) != 0:
                                for pindex, p in enumerate(self.model.trainable_variables):
                                    total_loss += self.MAS_weight / 2.0 * tf.reduce_sum(
                                        tf.convert_to_tensor(omegas[pindex], dtype=tf.float32) *
                                        (p - star_variables[pindex]) ** 2)

                            gradients = tape.gradient(total_loss, self.model.trainable_variables)
                            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                    if use_hard_buffer and len(hard_buffer) != 0:
                        xt = np.concatenate((x, xh))
                        yt = np.concatenate((y, yh))
                    else:
                        xt = x[:]
                        yt = y[:]

                    #yt_pred = self.model(tf.convert_to_tensor(xt, dtype=tf.float32))
                    yt_pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xt.T.tolist()])
                    #print(len(yt_pred))
                    accuracy = np.mean(np.argmax(yt_pred, axis=1) == yt)
                    msg += ' recent loss: {0:0.3f}'.format(np.mean(recent_loss.numpy()))

                    if use_hard_buffer and len(hard_buffer) != 0:
                        msg += ' hard loss: {0:0.3f}'.format(np.mean(hard_loss.numpy()))
                    losses.append(np.mean(accuracy))

                    loss_window.append(np.mean(first_train_loss))
                    if len(loss_window) > self.loss_window_length:
                        del loss_window[0]
                    loss_window_mean = np.mean(loss_window)
                    loss_window_variance = np.var(loss_window)

                    if not new_peak_detected and loss_window_mean > last_loss_window_mean + np.sqrt(
                            last_loss_window_variance):
                        new_peak_detected = True

                    if continual_learning and loss_window_mean < self.loss_window_mean_threshold and \
                            loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        count_updates += 1
                        update_tags.append(0.01)
                        last_loss_window_mean = loss_window_mean
                        last_loss_window_variance = loss_window_variance
                        new_peak_detected = False

                        gradients = [np.zeros_like(p.numpy()) for p in self.model.trainable_variables]

                        for sx in [_['state'] for _ in hard_buffer]:
                            with tf.GradientTape() as tape:
                                y_pred = self.model(tf.convert_to_tensor(np.asarray(sx).reshape(-1, self.dim), dtype=tf.float32))
                                loss = tf.norm(y_pred, ord=2, axis=1)
                            grads = tape.gradient(loss, self.model.trainable_variables)
                            for pindex, p in enumerate(grads):
                                gradients[pindex] += np.abs(p.numpy())

                        omegas_old = omegas[:]
                        omegas = []
                        star_variables = []

                        for pindex, p in enumerate(self.model.trainable_variables):
                            if len(omegas_old) != 0:
                                omegas.append(1 / count_updates * gradients[pindex] + (1 - 1 / count_updates) *
                                              omegas_old[pindex])
                            else:
                                omegas.append(gradients[pindex])
                            star_variables.append(p.numpy())

                    else:
                        update_tags.append(0)
                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)

                    if use_hard_buffer:
                        if len(hard_buffer) == 0:
                            loss = recent_loss.numpy()
                            #print(loss.shape,recent_loss.shape)
                        else:
                            loss = tf.concat([recent_loss, hard_loss], axis=0)
                            loss = loss.numpy()
                        
                        hard_buffer = []
                        #print(loss)
                        #loss = np.mean(loss)
                        sorted_inputs = [np.asarray(lx) for _, lx in reversed(sorted(zip(loss.tolist(), xt), key=lambda f: f[0]))]
                        sorted_targets = [ly for _, ly in reversed(sorted(zip(loss.tolist(), yt), key=lambda f: f[0]))]
                        
                        for i in range(min(self.hard_buffer_size, len(sorted_inputs))):
                            hard_buffer.append({'state': sorted_inputs[i],
                                                'trgt': sorted_targets[i]})
                        #return hard_buffer

                    #return data.test_inputs
                    for i in range(self.ntasks):
                        #y_pred = self.model.predict([tf.convert_to_tensor(np.asarray(data.test_inputs[i]))])
                        test_input = np.array([item for item in data.test_inputs[i]])

                        y_test__pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in test_input.T])
                        #print(test_input)
                        #y_pred = self.model(tf.convert_to_tensor(np.asarray(test_input), dtype=tf.float32))
                        y_sup = tf.one_hot(data.test_labels[i], depth=y_pred.shape[1], dtype=tf.float32)
                        test_accuracy = np.mean(
                            np.argmax(y_test__pred, axis=1) == data.test_labels[i])
                        print(test_accuracy)
                        test_loss[i].append(test_accuracy)
                        msg += ' test[{0}]: {1:0.3f}'.format(i, test_accuracy)
                        print('msg',msg)
                    #print('test loss',test_loss)
                    if self.verbose:
                        print(msg)

                    recent_buffer = []
                    update_times+=1
                    print('{0}th updating done'.format(update_times))

        if False and use_hard_buffer:
            xs_pos = [np.asarray(_['state'][0]) for _ in hard_buffer if _['trgt'] == 1]
            ys_pos = [np.asarray(_['state'][1]) for _ in hard_buffer if _['trgt'] == 1]
            xs_neg = [np.asarray(_['state'][0]) for _ in hard_buffer if _['trgt'] == 0]
            ys_neg = [np.asarray(_['state'][1]) for _ in hard_buffer if _['trgt'] == 0]
            plt.scatter(xs_pos, ys_pos, color='blue')
            plt.scatter(xs_neg, ys_neg, color='red')
            plt.title('replay buffer')
            plt.show()

        if False:
            for q in range(self.ntasks):
                y_pred = self.model(tf.convert_to_tensor(test_inputs[q].reshape(-1, self.dim), dtype=tf.float32)).numpy()
                positive_points = [test_inputs[q][i] for i in range(len(test_inputs[q])) if
                                   np.argmax(y_pred[i]) == 1]
                negative_points = [test_inputs[q][i] for i in range(len(test_inputs[q])) if
                                   np.argmax(y_pred[i]) == 0]
                plt.scatter([p[0] for p in positive_points], [p[1] for p in positive_points], color='blue')
                plt.scatter([p[0] for p in negative_points], [p[1] for p in negative_points], color='red')
            plt.axis('off')
            plt.show()

        print("duration: {0} minutes, count updates: {1}".format((time.time() - stime) / 60., count_updates))

        return losses, loss_window_means, update_tags, loss_window_variances, test_loss, future_losses