'''
Sketch2Normal implementation using Tensorflow for paper
Interactive Sketch-Based Normal Map Generation with Deep Neural Networks
Author Wanchao Su
This code is based on the implementation of pix2pix from  https://github.com/yenchenlin/pix2pix-tensorflow
'''
import os
import time
import random
import numpy as np
from glob import glob
import scipy.misc
from ops import *

class normalnet(object):
    def __init__(self, sess, image_size=256,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100, n_critic=5, clamp=0.01,
                 input_c_dim=3, output_c_dim=3, dataset_name='primitive', coefficient=100):

        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.clamp = clamp
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.coefficient = coefficient
        self.n_critic = n_critic

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.dataset_name = dataset_name
        self.build_model()

    def build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim + 1],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim+1:self.input_c_dim + self.output_c_dim+1]
        self.mask = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + 1]

        fake_B = self.generator(tf.concat([self.real_A, self.mask], axis=3))
        mask = tf.concat([self.mask, self.mask, self.mask], axis=3)
        self.fake_B = tf.add(tf.multiply(1-mask, fake_B), tf.multiply(mask, self.real_A))

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D_logits = self.discriminator(self.real_AB, reuse=False)
        self.D_logits_ = self.discriminator(self.fake_AB, reuse=True)

        self.fake_B_sample = self.generator(tf.concat([self.real_A, self.mask], axis=3), isSampling=True)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)
        self.real_B_sum = tf.summary.image("real_B", self.real_B)


        self.d_loss_real = tf.reduce_mean(self.D_logits)
        self.d_loss_fake = tf.reduce_mean(self.D_logits_)

        filter = 1.0/273*tf.constant([[1, 4, 7, 4, 1],
                                   [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7],
                                   [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]], tf.float32)

        filter = tf.reshape(filter, [5, 5, 1, 1])
        self.masked_loss = tf.reduce_mean(tf.multiply(tf.nn.conv2d(self.mask, filter, strides=[1, 1, 1, 1], padding='SAME'),
                                                      tf.reduce_sum(tf.abs(self.real_B - self.fake_B), axis=3)))
        self.pixel_wised_loss = tf.reduce_mean(tf.abs(self.real_B - self.fake_B))

        self.d_loss = self.d_loss_fake - self.d_loss_real
        self.g_loss = - self.d_loss_fake \
                      + self.L1_lambda * self.pixel_wised_loss \
                      + self.coefficient * self.L1_lambda * self.masked_loss

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.masked_loss_sum = tf.summary.scalar("masked_loss", self.masked_loss)
        self.pixel_wised_loss_sum = tf.summary.scalar("pixeled_loss", self.pixel_wised_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()


    def load_random_samples(self):
        data = np.random.choice(glob('./datasets/{}/val/*.png'.format(self.dataset_name)), self.batch_size)
        sample = [self.load_data(sample_file) for sample_file in data]
        sample_images = np.array(sample).astype(np.float32)
        return sample_images


    def sample_model(self, sample_dir, epoch, idx):
        sample_images = self.load_random_samples()
        samples, d_loss, g_loss = self.sess.run(
            [self.fake_B_sample, self.d_loss, self.g_loss],
            feed_dict={self.real_data: sample_images}
        )
        self.save_images(samples, [self.batch_size, 1], './{}/train_{:02d}_{:04d}.png'.format(sample_dir, epoch, idx))
        print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))

    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return h4

    def generator(self, image, isSampling=False):
        with tf.variable_scope("generator") as scope:

            if isSampling:
                scope.reuse_variables()

            s = self.output_size
            s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

            # image is (256 x 256 x 4)
            e1 = conv2d(image, self.gf_dim, name='g_e1_conv')
            e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'))
            e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'))
            e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'))
            e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'))
            e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'))
            e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'))
            e8 = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'))

            self.d1, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(e8),
                [self.batch_size, s128, s128, self.gf_dim*8], name='g_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(d1),
                [self.batch_size, s64, s64, self.gf_dim*8], name='g_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
                [self.batch_size, s32, s32, self.gf_dim*8], name='g_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_bn_d3(self.d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
                [self.batch_size, s16, s16, self.gf_dim*8], name='g_d4', with_w=True)
            d4 = self.g_bn_d4(self.d4)
            d4 = tf.concat([d4, e4], 3)

            self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
                [self.batch_size, s8, s8, self.gf_dim*4], name='g_d5', with_w=True)
            d5 = self.g_bn_d5(self.d5)
            d5 = tf.concat([d5, e3], 3)

            self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
                [self.batch_size, s4, s4, self.gf_dim*2], name='g_d6', with_w=True)
            d6 = self.g_bn_d6(self.d6)
            d6 = tf.concat([d6, e2], 3)

            self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
                [self.batch_size, s2, s2, self.gf_dim], name='g_d7', with_w=True)
            d7 = self.g_bn_d7(self.d7)
            d7 = tf.concat([d7, e1], 3)

            self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
                [self.batch_size, s, s, self.output_c_dim], name='g_d8', with_w=True)

            return tf.nn.tanh(self.d8)


    def save(self, checkpoint_dir, step):
        model_name = "sketch2normal.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def train(self, args):

        self.d_optim = tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.RMSPropOptimizer(learning_rate=args.lr).minimize(self.g_loss, var_list=self.g_vars)
        self.clip_d_vars_ops = [val.assign(tf.clip_by_value(val, -self.clamp, self.clamp)) for val in self.d_vars]
        tf.global_variables_initializer().run()

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_summary = tf.summary.merge([self.fake_B_sum, self.real_B_sum,self.d_loss_fake_sum, self.g_loss_sum])
        self.d_summary = tf.summary.merge([self.d_loss_real_sum, self.d_loss_sum])
        self.visual_loss_summary = tf.summary.merge([self.pixel_wised_loss_sum, self.masked_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(args.epoch):
            data = glob('./datasets/{}/train/*.png'.format(self.dataset_name))
            np.random.shuffle(data)
            batch_idxs = min(len(data), 1e8) // (self.batch_size*self.n_critic)
            print('[*] run optimizor...')

            for idx in xrange(0, batch_idxs):
                errD=.0
                batch_list = [self.load_training_imgs(data, idx+i) for i in xrange(self.n_critic)]
                for j in range(self.n_critic):
                    batch_images = batch_list[j]
                    _, errD, errd_real, errd_fake, errVis_sum, summary_str = self.sess.run([self.d_optim, self.d_loss,
                                                                                self.d_loss_real, self.d_loss_fake,
                                                                                            self.visual_loss_summary,
                                                                                            self.d_summary],
                                                                                           feed_dict={self.real_data: batch_images})
                    self.sess.run(self.clip_d_vars_ops)
                    self.writer.add_summary(summary_str, counter)
                    self.writer.add_summary(errVis_sum, counter)

                # Update G network
                _, errG, summary_str = self.sess.run([self.g_optim, self.g_loss, self.g_summary],
                                               feed_dict={self.real_data: batch_list[np.random.randint(0, self.n_critic, size=1)[0]]})
                self.writer.add_summary(summary_str, counter)

                current = time.time()
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs, current - start_time, errD, errG))
                start_time = current

                if np.mod(counter, 100) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, 1000) == 2:
                    self.save(args.checkpoint_dir, counter)
                counter += 1

    def test(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        test_files = glob('./datasets/{}/test/*.png'.format(self.dataset_name))

        n = [int(i) for i in map(lambda x: x.split('/')[-1].split('.png')[0], test_files)]
        test_files = [x for (y, x) in sorted(zip(n, test_files))]

        # load testing input
        print("Loading testing images ...")
        images = [self.load_data(file, is_test=True) for file in test_files]

        test_images = np.array(images).astype(np.float32)
        test_images = [test_images[i:i+self.batch_size] for i in xrange(0, len(test_images), self.batch_size)]
        test_images = np.array(test_images)
        print(test_images.shape)

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for i, test_image in enumerate(test_images):
            idx = i+1
            print("test image " + str(idx))
            results = self.sess.run(self.fake_B_sample, feed_dict={self.real_data: test_image})
            self.save_images(results, [self.batch_size, 1], './{}/test_{:04d}.png'.format(args.test_dir, idx))

    def load_training_imgs(self, data, idx):
        batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.load_data(batch_file) for batch_file in batch_files]

        batch_images = np.reshape(np.array(batch).astype(np.float32),
                                  (self.batch_size, self.image_size, self.image_size, -1))

        return batch_images

    def save_images(self, images, size, image_path):
        images = (images + 1) / 2.0
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        scipy.misc.imsave(image_path, img)

    def load_data(self, image_path, is_test=False):
        input_img = scipy.misc.imread(image_path).astype(np.float)
        index = image_path.find(self.dataset_name + '/')
        insert_point = index + len(self.dataset_name) + 1
        mask_path = image_path[:insert_point] + 'mask/' + image_path[insert_point:]
        mask = scipy.misc.imread(mask_path, mode='L').astype(np.float)
        img_A = input_img[:, 0:256, :]
        img_B = input_img[:, 256:512, :]

        if not is_test:
            num = random.randint(256, 286)
            img_A = scipy.misc.imresize(img_A, (num, num))
            img_B = scipy.misc.imresize(img_B, (num, num))
            mask = scipy.misc.imresize(mask, (num, num))
            num_1 = random.randint(0, num - 256)
            img_A = img_A[num_1:num_1 + 256, num_1:num_1 + 256, :]
            img_B = img_B[num_1:num_1 + 256, num_1:num_1 + 256, :]
            mask = mask[num_1:num_1 + 256, num_1:num_1 + 256]

        mask = np.reshape(mask, (256, 256, 1)) / 255.0
        img_A = img_A / 127.5 - 1.
        img_B = img_B / 127.5 - 1.
        img_AB = np.concatenate((img_A, mask, img_B), axis=2)

        return img_AB
