import tensorflow as tf
from ops import lrelu, conv2d, fully_connect, upscale, Pixl_Norm, downscale2d, MinibatchstateConcat
from utils import save_images
import numpy as np
from scipy.ndimage.interpolation import zoom
import math
import cv2
import random
class PGGAN(object):

    # build model
    def __init__(self,restore_model, restore_pg, image_size,hiding_size, batch_size, max_iters, model_path, read_model_path, data, sample_path, log_dir,
                 learn_rate, lam_gp, lam_eps, PG, t, use_wscale, is_celeba, lamda_adv):
        #1022
        self.restore_model = restore_model
        self.restore_pg = restore_pg
        self.image_size = image_size #can revise(also need to rewrite the PG)
        self.hiding_size = hiding_size
        self.lambda_recon = 1. - lamda_adv
        self.lambda_adv = lamda_adv
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.gan_model_path = model_path
        self.read_model_path = read_model_path
        self.data_In = data
        #self.sample_size = sample_size
        self.sample_path = sample_path
        self.log_dir = log_dir
        self.learning_rate = learn_rate
        self.lam_gp = lam_gp
        self.lam_eps = lam_eps
        self.pg = PG
        self.trans = t
        self.log_vars = []
        self.channel = self.data_In.channel
        self.output_size = 4 * pow(2, PG - 1)
        self.use_wscale = use_wscale
        self.is_celeba = is_celeba
        #self.images = tf.placeholder(tf.float32, [batch_size, self.image_size, self.image_size, self.channel])
        self.croppedImages = tf.placeholder( tf.float32, [batch_size, self.output_size, self.output_size, self.channel], name='croppedImages') #real images cropped
        self.fake_images = tf.placeholder( tf.float32, [batch_size, self.output_size, self.output_size, self.channel], name='fake_images') #inpainted images
        self.unfilledImages = tf.placeholder(tf.float32, [batch_size, self.image_size, self.image_size, self.channel], name='unfilledImages') #unfilled area
        # self.z = tf.placeholder(tf.float32, [self.batch_size, self.sample_size])
        self.alpha_tra = tf.Variable(initial_value=0.0, trainable=False,name='alpha_tra')
        

        lastencoderLayerNum = int(math.log(self.image_size) / math.log(2))
        lastencoderLayerNum = lastencoderLayerNum - 1 # minus 1 because the second last layer directly go from 4x4 to 1x1 
        print("lastencoderLayerNum=", lastencoderLayerNum)
        self.lastencoderLayerNum = lastencoderLayerNum

        lastdecoderLayerNum = int(math.log(self.hiding_size) / math.log(2))
        lastdecoderLayerNum = lastdecoderLayerNum - 1
        print("lastdecoderLayerNum=", lastdecoderLayerNum)
        self.lastdecoderLayerNum = lastdecoderLayerNum
        #1022
        

    def build_model_PGGan(self):

        self.fake_images = self.generate(cropsimage=self.unfilledImages, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.D_pro_logits = self.discriminate(self.croppedImages, reuse=False, pg = self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        _, self.G_pro_logits = self.discriminate(self.fake_images, reuse=True,pg= self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        
        mask_recon = tf.pad(tf.ones([self.output_size, self.output_size]), [[0,0], [0,0]])
        mask_recon = tf.reshape(mask_recon, [self.output_size, self.output_size, 1])
        mask_recon = tf.concat([mask_recon]*3,2)
        loss_recon_ori = tf.square( self.croppedImages - self.fake_images )
        loss_recon = tf.reduce_mean(tf.sqrt( 1e-5 + tf.reduce_sum(loss_recon_ori * mask_recon, [1,2,3]))) / 10.  # Loss for non-overlapping region
        
        #loss_recon = tf.square(self.croppedImages - self.fake_images )
        loss_adv_G = -tf.reduce_mean(self.G_pro_logits)
        # the defination of loss for D and G
        
        self.D_loss = tf.reduce_mean(self.G_pro_logits) - tf.reduce_mean(self.D_pro_logits)
        self.G_loss = loss_adv_G* self.lambda_adv + loss_recon * self.lambda_recon

        # gradient penalty from WGAN-GP
       
        self.differences = self.fake_images - self.croppedImages
        self.alpha = tf.random_uniform(shape=[self.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolates = self.croppedImages + (self.alpha * self.differences)
       
        _, discri_logits= self.discriminate(interpolates, reuse=True, pg=self.pg, t=self.trans, alpha_trans=self.alpha_tra)
        gradients = tf.gradients(discri_logits, [interpolates])[0]
     
        # 2 norm
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3]))
        self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        tf.summary.scalar("gp_loss", self.gradient_penalty)

        self.D_origin_loss = self.D_loss
        self.D_loss += self.lam_gp * self.gradient_penalty
        self.D_loss += self.lam_eps * tf.reduce_mean(tf.square(self.D_pro_logits))

        self.log_vars.append(("generator_loss", self.G_loss))
        self.log_vars.append(("discriminator_loss", self.D_loss))

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'dis' in var.name]

        total_para = 0
        for variable in self.d_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para += variable_para
        print ("The total para of D", total_para)

        self.g_vars = [var for var in t_vars if 'gen' in var.name]

        total_para2 = 0
        for variable in self.g_vars:
            shape = variable.get_shape()
            print (variable.name, shape)
            variable_para = 1
            for dim in shape:
                variable_para *= dim.value
            total_para2 += variable_para
        print ("The total para of G", total_para2)

        #save the variables , which remain unchanged
        self.d_vars_n = [var for var in self.d_vars if 'dis_n' in var.name]
        self.g_vars_n = [var for var in self.g_vars if 'gen_n' in var.name]   

        # remove the new variables for the new model
        self.d_vars_n_read = [var for var in self.d_vars_n if '{}'.format(self.output_size) not in var.name]
        if self.pg==1:
            self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(int(self.output_size/4)) not in var.name]
        else:
            self.g_vars_n_read = [var for var in self.g_vars_n if '{}'.format(int(self.output_size/2)) not in var.name]

        # save the rgb variables, which remain unchanged
        self.d_vars_n_2 = [var for var in self.d_vars if 'dis_y_rgb_conv' in var.name]
        self.g_vars_n_2 = [var for var in self.g_vars if 'gen_other' in var.name]

        self.d_vars_n_2_rgb = [var for var in self.d_vars_n_2 if '{}'.format(self.output_size) not in var.name]
        self.g_vars_n_2_rgb = [var for var in self.g_vars_n_2 if '{}'.format(self.pg) not in var.name]

        print ("d_vars", len(self.d_vars))
        print ("g_vars", len(self.g_vars))

        print ("self.d_vars_n_read", len(self.d_vars_n_read))
        print ("self.g_vars_n_read", len(self.g_vars_n_read))


        print ("d_vars_n_2_rgb", len(self.d_vars_n_2_rgb))
        print ("g_vars_n_2_rgb", len(self.g_vars_n_2_rgb))

        # for n in self.d_vars:
        #     print (n.name)

        self.g_d_w = [var for var in self.d_vars + self.g_vars if 'bias' not in var.name]

        print ("self.g_d_w", len(self.g_d_w))

        self.saver = tf.train.Saver(self.d_vars + self.g_vars)
        self.r_saver = tf.train.Saver(self.d_vars_n_read + self.g_vars_n_read)

        if len(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb):
            self.rgb_saver = tf.train.Saver(self.d_vars_n_2_rgb + self.g_vars_n_2_rgb)
            
        for k, v in self.log_vars:
            tf.summary.scalar(k, v)

    # do train
    def train(self):
        step_pl = tf.placeholder(tf.float32, shape=None)
        alpha_tra_assign = self.alpha_tra.assign(step_pl / self.max_iters)

        opti_D = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.D_loss, var_list=self.d_vars)
        opti_G = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.0, beta2=0.99).minimize(
            self.G_loss, var_list=self.g_vars)

        init = tf.global_variables_initializer()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(init)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)
            if self.pg != 1 and self.pg != 7:
                if self.trans:
                    print(self.read_model_path+"model_final.ckpt")
                    self.r_saver.restore(sess, self.read_model_path+"model_final.ckpt")
                    self.rgb_saver.restore(sess, self.read_model_path+"model_final.ckpt")

                else:
                    self.saver.restore(sess, self.read_model_path+"model_final.ckpt")

            step = 0
            batch_num = 0
            while step <= self.max_iters:
                # optimization D
                n_critic = 1
                if self.pg >= 5:
                    n_critic = 1

                for i in range(n_critic):
                   # sample_z = np.random.normal(size=[self.batch_size, self.sample_size])
                    crop_pos = (self.image_size - self.hiding_size)/2
                    if self.is_celeba:
                        train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                        images_array = self.data_In.getShapeForData(train_list, resize_w=self.image_size)
                        unfilledImages, crops,rand_x,rand_y= self.crop_random(images_array, width=self.hiding_size, height=self.hiding_size, x=crop_pos, y=crop_pos, isShake=False)
                        #imagescrops, crops,_,_ = zip(*images_crops)
                       
                        #realbatch_array = self.data_In.getShapeForData(crops, resize_w=self.output_size)
                        realbatch_array = crops
                    else:
                        realbatch_array = self.data_In.getNextBatch(self.batch_size, resize_w=self.output_size)
                        realbatch_array = np.transpose(realbatch_array, axes=[0, 3, 2, 1]).transpose([0, 2, 1, 3])

                    if self.trans and self.pg != 0:
                        alpha = np.float(step) / self.max_iters
                        low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
                        low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')
                        realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

                    resahpe_realbatch_array = np.zeros([realbatch_array.shape[0], self.croppedImages.shape[1], self.croppedImages.shape[2], realbatch_array.shape[3]])
                    for i in range(realbatch_array.shape[0]):
                        resahpe_realbatch_array[i, :, :, :] = cv2.resize(realbatch_array[i], (self.croppedImages.shape[1], self.croppedImages.shape[2]))
                    realbatch_array = resahpe_realbatch_array
                    reshape_unfilledImages = np.zeros([unfilledImages.shape[0], self.unfilledImages.shape[1], self.unfilledImages.shape[2], unfilledImages.shape[3]])

                    sess.run(opti_D, feed_dict={self.croppedImages: realbatch_array, self.unfilledImages: unfilledImages})
                    batch_num += 1

                # optimization G
                sess.run(opti_G, feed_dict={self.croppedImages: realbatch_array,self.unfilledImages: unfilledImages})


                summary_str = sess.run(summary_op, feed_dict={self.croppedImages: realbatch_array, self.unfilledImages: unfilledImages})
                summary_writer.add_summary(summary_str, step)
                summary_writer.add_summary(summary_str, step)
                # the alpha of fake_in process
                sess.run(alpha_tra_assign, feed_dict={step_pl: step})

                if step % 1000 == 0:
                    D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.croppedImages: realbatch_array,self.unfilledImages: unfilledImages})
                    print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

                    realbatch_array = np.clip(realbatch_array, -1, 1)
                    save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2],
                                '{}/{:02d}_real.jpg'.format(self.sample_path, step))
                    images_array = np.clip(images_array, -1, 1)
                    save_images(images_array[0:self.batch_size], [2, self.batch_size/2],
                                '{}/{:02d}_uncropped.jpg'.format(self.sample_path, step))
                    if self.trans and self.pg != 0:
                        low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
                        save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
                                    '{}/{:02d}_real_lower.jpg'.format(self.sample_path, step))
                    print("=====rand_x, rand_y : "+str(rand_x)+", " + str(rand_y))
                    fake_image = sess.run(self.fake_images,
                                          feed_dict={self.croppedImages: realbatch_array, self.unfilledImages: unfilledImages})
                    fake_image = np.clip(fake_image, -1, 1)
                    save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train.jpg'.format(self.sample_path, step))
                    inpainted_images = images_array
                    inpainted_images[:, int(rand_y):int(rand_y+fake_image.shape[1]), int(rand_x):int(rand_x+fake_image.shape[2]),:] = fake_image
                    save_images(inpainted_images[0:self.batch_size], [2, self.batch_size/2], '{}/{:02d}_train_inpainted.jpg'.format(self.sample_path, step))
                if np.mod(step, 1500) == 0 and step != 0:
                    self.saver.save(sess, self.gan_model_path+"model_{}.ckpt".format(step))

                step += 1

            save_path = self.saver.save(sess, self.gan_model_path+"model_final.ckpt")
            print ("Model saved in file: %s" % save_path)

        tf.reset_default_graph()
    def discriminate(self, conv, reuse=False, pg=1, t=False, alpha_trans=0.01):
        #dis_as_v = []
        with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE) as scope:
            if reuse == True:
                scope.reuse_variables()
            if t:
                conv_iden = downscale2d(conv)
                #from RGB
                conv_iden = lrelu(conv2d(conv_iden, output_dim= self.get_nf(pg - 2), k_w=1, k_h=1, d_h=1, d_w=1, use_wscale=self.use_wscale,name='dis_y_rgb_conv_{}'.format(conv_iden.shape[1])))
            # fromRGB
           
            conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1), k_w=1, k_h=1, d_w=1, d_h=1, use_wscale=self.use_wscale, name='dis_y_rgb_conv_{}'.format(conv.shape[1])))

            for i in range(pg - 1):
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 1 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                    name='dis_n_conv_1_{}'.format(conv.shape[1])))
                conv = lrelu(conv2d(conv, output_dim=self.get_nf(pg - 2 - i), d_h=1, d_w=1, use_wscale=self.use_wscale,
                                                      name='dis_n_conv_2_{}'.format(conv.shape[1])))
                conv = downscale2d(conv)
                if i == 0 and t:
                    conv = alpha_trans * conv + (1 - alpha_trans) * conv_iden

            conv = MinibatchstateConcat(conv)
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=3, k_h=3, d_h=1, d_w=1, use_wscale=self.use_wscale, name='dis_n_conv_1_{}'.format(conv.shape[1])))
            conv = lrelu(
                conv2d(conv, output_dim=self.get_nf(1), k_w=4, k_h=4, d_h=1, d_w=1, use_wscale=self.use_wscale, padding='VALID', name='dis_n_conv_2_{}'.format(conv.shape[1])))
            conv = tf.reshape(conv, [self.batch_size, -1])

            #for D
            output = fully_connect(conv, output_size=1, use_wscale=self.use_wscale, gain=1, name='dis_n_fully')

            return tf.nn.sigmoid(output), output

    def generate(self, cropsimage, pg=1, t=False, alpha_trans=0.0):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE) as scope:
            #batchx256x256x3
            conv1 = self.new_conv_layer(cropsimage, [4,4,3,64], stride=2,name='gen_n_oneconv')
            bn1 = lrelu(self.batchnorm(conv1))
            #batchx128x128x64
            conv2 = self.new_conv_layer(bn1, [4,4,64,64], stride=2, name='gen_n_twoconv'.format(bn1.shape[1]))
            bn2 = lrelu(self.batchnorm(conv2))
            #batchx64x64x64
            conv3 = self.new_conv_layer(bn2, [4,4,64,128], stride=2, name='gen_n_threeconv'.format(bn2.shape[1]))
            bn3 = lrelu(self.batchnorm(conv3))
            #batchx32x32x128
            conv4 = self.new_conv_layer(bn3, [4,4,128,256], stride=2, name='gen_n_fourconv'.format(bn3.shape[1]))
            bn4 = lrelu(self.batchnorm(conv4))
            #batchx16x16x256
            conv5 = self.new_conv_layer(bn4, [4,4,256,512], stride=2,  name='gen_n_fiveconv'.format(bn4.shape[1]))
            bn5 = lrelu(self.batchnorm(conv5))
            #batchx8x8x512
            pool = tf.nn.max_pool(bn5, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')
            #batchx4x4x512
            conv6 = self.new_conv_layer(pool, [4,4,512,4000],stride=2, padding='VALID', name='gen_n_sixconv'.format(pool.shape[1]))
            bn6 = lrelu(self.batchnorm(conv6))
            #batchx1x1x4000

            previousFeatureMap = bn6
            featureMapSize = 4
            previousDepth = 4000 
            depth = 512
            for i in range(pg,0,-1):
                if i == pg:
                    deconv = self.new_deconv_layer( previousFeatureMap, [4,4,depth,previousDepth], [self.batch_size,featureMapSize,featureMapSize,depth],padding='VALID',stride=2,name='gen_n_deconv_{}'.format(previousFeatureMap.shape[1]))#batchx4x4x512
                else:
                    deconv = self.new_deconv_layer( previousFeatureMap, [4,4,depth,previousDepth], [self.batch_size,featureMapSize,featureMapSize,depth],padding='SAME',stride=2, name='gen_n_deconv_{}'.format(previousFeatureMap.shape[1]))

                debn = tf.nn.relu(self.batchnorm(deconv))
                previousFeatureMap = debn
                previousDepth = depth
                depth = int(depth / 2)
                featureMapSize = featureMapSize *2
            #print("pg:"+str(pg))
            #print("previousFeatureMap:"+str(previousFeatureMap))

            if pg == 1:
                recon = self.new_deconv_layer( previousFeatureMap, [4,4,3,previousDepth], [self.batch_size,self.output_size,self.output_size,3],stride=1,padding='SAME',name='gen_other_{}'.format(pg))
                #batchx4x4x3
            else:
                recon = self.new_deconv_layer( previousFeatureMap, [4,4,3,previousDepth], [self.batch_size,self.output_size,self.output_size,3],stride=1,padding='SAME',name='gen_other_{}'.format(pg))
                #batchxOutputxOutputx3

        #if t: recon = (1 - alpha_trans) * de_iden + alpha_trans*recon(fade)
        return recon
    
    def get_nf(self, stage):
        return min(1024 / (2 **(stage * 1)), 512)
    
    def sample_z(self, mu, log_var):
        eps = tf.random_normal(shape=tf.shape(mu))
        return mu + tf.exp(log_var / 2) * eps

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, bottom, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]
        normed = tf.contrib.layers.batch_norm(bottom, decay=0.5, epsilon=epsilon, scale=False)


        return normed

    def crop_random(self,image_ori, width=64,height=64, x=None, y=None, isShake = False):
        overlap=32
        if image_ori is None: return None

        random_y = 32 if x is None else x
        random_x = 32 if y is None else y

        if isShake:
            random_y = np.random.randint(0,127) 
            random_x = np.random.randint(0,127) 
        
        image = image_ori.copy()
        crop = image_ori.copy()
        crop = crop[:,int(random_y):int(random_y+height), int(random_x):int(random_x+width),:]

        image[:,int(random_y) :int(random_y+height) , int(random_x) :int(random_x+width) , 0] = 123.68 / 255.0
        image[:,int(random_y) :int(random_y+height) , int(random_x) :int(random_x+width) , 1] = 116.779 / 255.0
        image[:,int(random_y) :int(random_y+height) , int(random_x) :int(random_x+width) , 2] = 103.939 / 255.0


        return image, crop, random_x, random_y


    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name="conv" ):
        with tf.variable_scope(name) as scope:
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, bottom, filter_shape, output_shape, activation=tf.identity, padding='SAME', stride=1, name="deconv"):
        with tf.variable_scope(name)as scope:
            W = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-2],
                    initializer=tf.constant_initializer(0.))
            deconv = tf.nn.conv2d_transpose( bottom, W, output_shape, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(deconv, b))

        return bias

    def test(self, batch_num=None, test_image=None):
        fake = tf.identity(self.fake_images)
        init = tf.global_variables_initializer()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            sess.run(init)
            self.saver.restore(sess, self.read_model_path)
            if batch_num is None:
                batch_num = random.randint(0,10000)

            # optimization D

            crop_pos = (self.image_size - self.hiding_size)/2
            if self.is_celeba:
                train_list = self.data_In.getNextBatch(batch_num, self.batch_size)
                images_array = self.data_In.getShapeForData(train_list, resize_w=self.image_size)
                unfilledImages, crops,rand_x,rand_y= self.crop_random(images_array, width=self.hiding_size, height=self.hiding_size, x=crop_pos, y=crop_pos, isShake=False)
                
                realbatch_array = crops
            else:
                realbatch_array = self.data_In.getNextBatch(self.batch_size, resize_w=self.output_size)
                realbatch_array = np.transpose(realbatch_array, axes=[0, 3, 2, 1]).transpose([0, 2, 1, 3])

            #if self.trans and self.pg != 0:
            #    alpha = np.float(step) / self.max_iters
            #    low_realbatch_array = zoom(realbatch_array, zoom=[1, 0.5, 0.5, 1], mode='nearest')
            #    low_realbatch_array = zoom(low_realbatch_array, zoom=[1, 2, 2, 1], mode='nearest')
            #    realbatch_array = alpha * realbatch_array + (1 - alpha) * low_realbatch_array

            resahpe_realbatch_array = np.zeros([realbatch_array.shape[0], self.croppedImages.shape[1], self.croppedImages.shape[2], realbatch_array.shape[3]])
            for i in range(realbatch_array.shape[0]):
                resahpe_realbatch_array[i, :, :, :] = cv2.resize(realbatch_array[i], (self.croppedImages.shape[1], self.croppedImages.shape[2]))
            realbatch_array = resahpe_realbatch_array
            reshape_unfilledImages = np.zeros([unfilledImages.shape[0], self.unfilledImages.shape[1], self.unfilledImages.shape[2], unfilledImages.shape[3]])

            #sess.run(opti_D, feed_dict={self.croppedImages: realbatch_array, self.unfilledImages: unfilledImages})
            #batch_num += 1

            # optimization G
            #sess.run(opti_G, feed_dict={self.unfilledImages: unfilledImages})

            #sess.run(alpha_tra_assign, feed_dict={step_pl: step})

            #D_loss, G_loss, D_origin_loss, alpha_tra = sess.run([self.D_loss, self.G_loss, self.D_origin_loss,self.alpha_tra], feed_dict={self.croppedImages: realbatch_array,self.unfilledImages: unfilledImages})
            #print("PG %d, step %d: D loss=%.7f G loss=%.7f, D_or loss=%.7f, opt_alpha_tra=%.7f" % (self.pg, step, D_loss, G_loss, D_origin_loss, alpha_tra))

            realbatch_array = np.clip(realbatch_array, -1, 1)
            save_images(realbatch_array[0:self.batch_size], [2, self.batch_size/2],
                        '{}/real.jpg'.format(self.sample_path))
            images_array = np.clip(images_array, -1, 1)
            save_images(images_array[0:self.batch_size], [2, self.batch_size/2],
                        '{}/uncropped.jpg'.format(self.sample_path))
            #if self.trans and self.pg != 0:
            #    low_realbatch_array = np.clip(low_realbatch_array, -1, 1)
            #    save_images(low_realbatch_array[0:self.batch_size], [2, self.batch_size / 2],
            #                '{}/{:02d}_real_lower.jpg'.format(self.sample_path, step))

            print("=====rand_x, rand_y : "+str(rand_x)+", " + str(rand_y))
            fake_image = sess.run(fake,
                                    feed_dict={self.croppedImages: realbatch_array, self.unfilledImages: unfilledImages})
            fake_image = np.clip(fake_image, -1, 1)
            save_images(fake_image[0:self.batch_size], [2, self.batch_size/2], '{}/train.jpg'.format(self.sample_path))
            inpainted_images = images_array
            inpainted_images[:, int(rand_y):int(rand_y+fake_image.shape[1]), int(rand_x):int(rand_x+fake_image.shape[2]),:] = fake_image
            save_images(inpainted_images[0:self.batch_size], [2, self.batch_size/2], '{}/train_inpainted.jpg'.format(self.sample_path))

        tf.reset_default_graph()
 



