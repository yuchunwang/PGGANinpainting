import tensorflow as tf

from utils import mkdir_p
from PGGAN256 import PGGAN
from utils import CelebA, CelebA_HQ
flags = tf.app.flags
import os

os.environ['CUDA_VISIBLE_DEVICES']='0'

flags.DEFINE_string("OPER_NAME", "PretrainedCheat_iter5k_lr2_adv0.2", "the name of experiments")
flags.DEFINE_integer("OPER_FLAG", 0, "Flag of opertion: 0 is for training ")
flags.DEFINE_string("path",'?', "Path of training data, for example /home/hehe/")
flags.DEFINE_integer("batch_size", 16, "Batch size")
# flags.DEFINE_integer("sample_size"512, "Size of sample")
flags.DEFINE_integer("image_size", 256, "input image size")
flags.DEFINE_integer("hiding_size", 128, "Maxmization inpainting size")
flags.DEFINE_integer("max_iters", 5000, "Maxmization of training number")
flags.DEFINE_float("learn_rate", 0.002, "Learning rate for G and D networks")
flags.DEFINE_integer("lam_gp", 10, "Weight of gradient penalty term")
flags.DEFINE_float("lam_eps", 0.001, "Weight for the epsilon term")
flags.DEFINE_float("lamda_adv", 0.2, "lamda adv")

flags.DEFINE_integer("flag", 11, "FLAG of gan training process")
flags.DEFINE_boolean("use_wscale", True, "Using the scale of weight")
flags.DEFINE_boolean("celeba", True, "Whether using celeba or using CelebA-HQ")

flags.DEFINE_string("restore_model",None, "Keep trained ckpt")
flags.DEFINE_integer("restore_pg", None,"Keep trained pg size start")

FLAGS = flags.FLAGS
if __name__ == "__main__":

    root_log_dir = "./output/{}/logs/".format(FLAGS.OPER_NAME)
    mkdir_p(root_log_dir)

    if FLAGS.celeba:
        data_In = CelebA(FLAGS.path)
    else:
        data_In = CelebA_HQ(FLAGS.path)

    print ("the num of dataset", len(data_In.image_list))

    if FLAGS.OPER_FLAG == 0:
        #fl = [2,3,3,4,4,5,5,6,6,7,7]
        #r_fl = [2,2,3,3,4,4,5,5,6,6,7]
        fl = [6,6]
        r_fl = [5,6]
        for i in range(len(fl)):

            t = False if (i % 2 == 1) else True
            pggan_checkpoint_dir_write = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i])
            sample_path = "./output/{}/{}/sample_{}_{}".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, fl[i], t)
            mkdir_p(pggan_checkpoint_dir_write)
            mkdir_p(sample_path)
            pggan_checkpoint_dir_read = "./output/{}/model_pggan_{}/{}/".format(FLAGS.OPER_NAME, FLAGS.OPER_FLAG, r_fl[i])

            pggan = PGGAN(restore_model=FLAGS.restore_model, restore_pg=FLAGS.restore_pg, image_size=FLAGS.image_size,batch_size=FLAGS.batch_size, hiding_size=FLAGS.hiding_size, 
            max_iters=FLAGS.max_iters, model_path=pggan_checkpoint_dir_write, read_model_path=pggan_checkpoint_dir_read,data=data_In,sample_path=sample_path, log_dir=root_log_dir, 
            learn_rate=FLAGS.learn_rate, lam_gp=FLAGS.lam_gp, lam_eps=FLAGS.lam_eps, PG= fl[i],t=t, use_wscale=FLAGS.use_wscale, is_celeba=FLAGS.celeba, lamda_adv=FLAGS.lamda_adv)
            
            pggan.build_model_PGGan()
            pggan.train()



