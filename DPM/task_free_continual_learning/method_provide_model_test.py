import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.optimizers import Nadam

tf.get_logger().setLevel('ERROR')


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
                loss_window_length=None,
                loss_window_mean_threshold=None,
                loss_window_variance_threshold=None, 
                MAS_weight=None,
                recent_buffer_size=50,
                hard_buffer_size=50,
                model=None):
        # Define your model architecture here
        self.model = model
        self.optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
                epsilon=1e-08, clipvalue=3)
        # self.optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
        #self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.loss_fn = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

        # Initialize an Adam optimizer
        # self.optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999,
        #         epsilon=1e-08, schedule_decay=0.004, clipvalue=3)

        # Initialize a categorical cross-entropy loss function
        # self.loss_fn = CategoricalCrossentropy()

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
        prediction_results = {}
        prediction_labels = []
        actual_labels = []

        for t in range(self.ntasks):
            for s in range(len(data.test_inputs[t][500:])):
                # Initialize an empty list to store the prediction results and actual labels
                recent_buffer.append({'state': data.test_inputs[t][s], 'trgt': data.test_labels[t][s]})
                if len(recent_buffer) > self.recent_buffer_size:
                    del recent_buffer[0]

                if len(recent_buffer) == self.recent_buffer_size:
                    msg = 'task: {0} step: {1}'.format(t, s)
                    #print('recent buffer is full',s)
                    #return recent_buffer
                    x = np.asarray([_['state'] for _ in recent_buffer])
                    y = np.asarray([_['trgt'] for _ in recent_buffer])
                    #return x,y
                    #test on future data
                    xf=x[:]
                    yf=y[:]
                    yf_pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xf.T.tolist()])
                    # Inside the loop where you calculate the accuracy
                    prediction_labels.extend(np.argmax(yf_pred, axis=1).tolist())
                    actual_labels.extend(yf.tolist())
                    accuracy = np.mean(np.argmax(yf_pred, axis=1) == yf)
                    future_losses.append(accuracy)

                    if use_hard_buffer and len(hard_buffer) != 0:
                        print('hard buffer is not empty',s)
                        xh = np.asarray([_['state'] for _ in hard_buffer])
                        yh = np.asarray([_['trgt'] for _ in hard_buffer])

                    # # with batch
                    # batch_size = 32
                    # # Calculate the number of batches per epoch
                    # num_batches = len(xf) // batch_size
                    # for gs in range(self.gradient_steps):
                    #     for i in range(num_batches):
                    #         # Get the current batch of data
                    #         x = xf[i*batch_size:(i+1)*batch_size]
                    #         y = yf[i*batch_size:(i+1)*batch_size]
                    #         with tf.GradientTape() as tape:

                    #             y_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in x.T.tolist()],training=True)
                    #             y_sup = tf.one_hot(tf.convert_to_tensor(y, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                    #             recent_loss = self.loss_fn(y_sup, y_pred)
                    #             total_loss = tf.reduce_sum(recent_loss)
                    #             #print('recent loss',recent_loss,'total loss',total_loss)
                    #             if use_hard_buffer and len(hard_buffer) != 0:
                    #                 yh_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xh.T],training=True)
                    #                 yh_sup = tf.one_hot(tf.convert_to_tensor(yh, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                    #                 hard_loss = self.loss_fn(yh_sup, yh_pred)
                    #                 total_loss += tf.reduce_sum(hard_loss)
                    #                 #print('hard loss', hard_loss,'total loss', total_loss)

                    #             if gs == 0:
                    #                 first_train_loss = total_loss.numpy()

                    #             if continual_learning and len(star_variables) != 0 and len(omegas) != 0:
                    #                 #print('add MAS regularization')
                    #                 for pindex, p in enumerate(self.model.trainable_variables):
                    #                     total_loss += self.MAS_weight / 2.0 * tf.reduce_sum(
                    #                         tf.convert_to_tensor(omegas[pindex], dtype=tf.float32) *
                    #                         (p - star_variables[pindex]) ** 2)

                    #             gradients = tape.gradient(total_loss, self.model.trainable_variables)
                    #             self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    for gs in range(self.gradient_steps):
                        with tf.GradientTape() as tape:

                            y_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in x.T.tolist()],training=True)
                            y_sup = tf.one_hot(tf.convert_to_tensor(y, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                            recent_loss = self.loss_fn(y_sup, y_pred)
                            total_loss = tf.reduce_sum(recent_loss)
                            #print('recent loss',recent_loss,'total loss',total_loss)
                            if use_hard_buffer and len(hard_buffer) != 0:
                                yh_pred = self.model([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in xh.T],training=True)
                                yh_sup = tf.one_hot(tf.convert_to_tensor(yh, dtype=tf.int32), depth=y_pred.shape[1], dtype=tf.float32)
                                hard_loss = self.loss_fn(yh_sup, yh_pred)
                                total_loss += tf.reduce_sum(hard_loss)
                                #print('hard loss', hard_loss,'total loss', total_loss)

                            if gs == 0:
                                first_train_loss = total_loss.numpy()

                            if continual_learning and len(star_variables) != 0 and len(omegas) != 0:
                                #print('add MAS regularization')
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

                    # add loss to loss_window and detect loss plateaus
                    loss_window.append(np.mean(first_train_loss))
                    #print('first train loss', first_train_loss,'loss window',loss_window)

                    if len(loss_window) > self.loss_window_length:
                        del loss_window[0]
                    loss_window_mean = np.mean(loss_window)
                    loss_window_variance = np.var(loss_window)
                    #print('loss_window_mean',loss_window_mean,'loss_window_variance',loss_window_variance)

                    if not new_peak_detected and loss_window_mean > last_loss_window_mean + np.sqrt(
                            last_loss_window_variance):
                        new_peak_detected = True

                    if continual_learning and loss_window_mean < self.loss_window_mean_threshold and \
                            loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        print('start updating importance weights')
                        #break
                        count_updates += 1
                        update_tags.append(0.01)
                        last_loss_window_mean = loss_window_mean
                        last_loss_window_variance = loss_window_variance
                        # new_peak_detected = False

                        gradients = [np.zeros_like(p.numpy()) for p in self.model.trainable_variables]

                        for sx in [_['state'] for _ in hard_buffer]:
                            #print(type(sx),sx)
                            with tf.GradientTape() as tape:
                                y_pred = self.model([tf.convert_to_tensor([x]) for x in sx])
                                # y_pred = self.model(tf.convert_to_tensor(np.array(sx).reshape(1, -1), dtype=tf.float32))
                                #y_pred = self.model(tf.convert_to_tensor(np.array(sx).reshape(1,-1), dtype=tf.float32))
                                loss = tf.norm(y_pred, ord=2, axis=1)
                            grads = tape.gradient(loss, self.model.trainable_variables)
                            for pindex, p in enumerate(grads):
                                if isinstance(p, tf.IndexedSlices):
                                    p = tf.convert_to_tensor(p)
                                gradients[pindex] += np.abs(p.numpy())

                            # with tf.GradientTape() as tape:
                            #     y_pred = self.model(tf.convert_to_tensor(np.asarray(sx).reshape(-1, 1), dtype=tf.float32))
                            #     loss = tf.norm(y_pred, ord=2, axis=1)
                            # grads = tape.gradient(loss, self.model.trainable_variables)
                            # for pindex, p in enumerate(grads):
                            #     gradients[pindex] += np.abs(p.numpy())

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
                            # print('omegas:',len(omegas),omegas)
                            # print('star_variables:',len(star_variables),star_variables)
                        # print('omegas:',len(omegas))
                        # print('star_variables:',len(star_variables))

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
                        # print(type(loss))
                        # print(loss)
                        # print(type(xt))
                        # print(xt)
                        sorted_inputs = [np.asarray(lx) for _, lx in reversed(sorted(zip(loss.tolist(), xt), key=lambda f: f[0]))]
                        sorted_targets = [ly for _, ly in reversed(sorted(zip(loss.tolist(), yt), key=lambda f: f[0]))]
                        
                        for i in range(min(self.hard_buffer_size, len(sorted_inputs))):
                            hard_buffer.append({'state': sorted_inputs[i],
                                                'trgt': sorted_targets[i]})
                        #return hard_buffer

                    # #return data.test_inputs
                    # for i in range(self.ntasks):
                    #     #y_pred = self.model.predict([tf.convert_to_tensor(np.asarray(data.test_inputs[i]))])
                    #     test_input = np.array([item for item in data.test_inputs[i]])

                    #     y_test__pred = self.model.predict([tf.convert_to_tensor(np.asarray(row).reshape(-1,1)) for row in test_input.T])
                    #     #print(test_input)
                    #     #y_pred = self.model(tf.convert_to_tensor(np.asarray(test_input), dtype=tf.float32))
                    #     y_sup = tf.one_hot(data.test_labels[i], depth=y_pred.shape[1], dtype=tf.float32)
                    #     test_accuracy = np.mean(
                    #         np.argmax(y_test__pred, axis=1) == data.test_labels[i])
                    #     #print(test_accuracy)
                    #     test_loss[i].append(test_accuracy)
                    #     msg += ' test[{0}]: {1:0.3f}'.format(i, test_accuracy)
                    #     print('msg',msg)
                    # #print('test loss',test_loss)
                    # if self.verbose:
                    #     print(msg)

                    recent_buffer = []
                    update_times+=1
                    print('{0}th updating done'.format(update_times))


        prediction_results['actual_labels'] = actual_labels
        prediction_results['prediction_labels'] = prediction_labels
        print("duration: {0} minutes, count updates: {1}".format((time.time() - stime) / 60., count_updates))

        return losses, loss_window_means, update_tags, loss_window_variances, future_losses, prediction_results