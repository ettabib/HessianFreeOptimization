""" Hessian Free Optimizer """
""" Author: MoonLight, 2018 """


import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class clr:
  """ Used for color debug output to console. """
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'


class HFOptimizer(object):
  """ Tensorflow based Hessian-Free (Truncated Newton) optimizer.
  More details: (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011).

  Methods to use:
  __init__:
      Creates Tensorflow graph and variables.
  minimize:
      Perfoms HF optimization. """

  DUMPING_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-4
  CG_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-20
  DUMPING_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-8
  CG_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-80

  def __init__(self, loss, model,
                learning_rate=1,
                cg_decay=0.95,
                damping=0.5,
                adjust_damping=True,
                batch_size=None,
                use_gauss_newton_matrix=False,
                preconditioner=False,
                prec_loss=None,
                gap=10,
                cg_max_iters=50,
                dtype=tf.float32):
      # ... (keep the existing parameter descriptions as is)

    self.loss = loss
    self.cg_decay = cg_decay
    self.prec_loss = prec_loss
    self.batch_size = batch_size
    self.use_prec = preconditioner
    self.learning_rate = learning_rate
    self.use_gnm = use_gauss_newton_matrix
    self.damping = damping
    self.gap = gap
    self.cg_max_iters = cg_max_iters
    self.adjust_damping = adjust_damping
    self.damp_var = tf.Variable(initial_value=0, dtype=dtype)
    self.model = model
    self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT32
    self.damp_num_err = HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT64
    if dtype == tf.float64:
        self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT64
        self.damp_num_err = HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT64
    if not self.use_gnm:
        self.damp_num_err = 1e-1

    if not self.use_gnm and self.use_prec:
        self.use_prec = False
        print(clr.WARNING + 'WARNING: You set preconditioner to True but ' +
              'use_gauss_newton_matrix to False, and it\'s prohibited, so we set ' +
              'preconditioner back to False, if you ask why see more information ' +
              'on (Martens & Sutskever, ICML 2011).' + clr.ENDC)
    elif self.use_prec and self.use_gnm and self.prec_loss is None:
        print(clr.WARNING + 'WARNING: If you use preconditioner it is ' +
              'better to set prec_loss explicitly, because it can cause graph ' +
              'making problem. (What\'s prec_loss see in description)' + clr.ENDC)

    self.W = self.model.trainable_variables

    with tf.name_scope('cg_vars'):
        self.cg_step = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.cg_delta = []
        self.directions = []
        self.residuals = []
        for w in self.W:
            zeros = tf.zeros(w.shape, dtype=dtype)
            delta = tf.Variable(zeros, dtype=dtype, name='delta')
            self.cg_delta.append(delta)
            d = tf.Variable(zeros, dtype=dtype, name='direction')
            self.directions.append(d)
            r = tf.Variable(zeros, dtype=dtype, name='residual')
            self.residuals.append(r)

    with tf.GradientTape() as tape:
        tape.watch(self.W)
        loss_value = self.loss(self.model.predict(self.model.inputs), self.model.targets)
    self.grads = tape.gradient(loss_value, self.W)

    cg_op, res_norm, dl = self.__conjugate_gradient(self.grads)
    self.ops = {
        'cg_update': cg_op,
        'res_norm': res_norm,
        'dl': dl,
        'set_delta_0': self.__update_delta_0(),
        'train': self.__train_op()
    }
    
  def minimize(self, model, debug_print=False):
      self.cg_step.assign(0)

      if self.adjust_damping:
          loss_before_cg = self.loss(model.predict(model.inputs))

      dl_track = [self.ops['dl'](model)]
      self.ops['set_delta_0']()

      for i in range(self.cg_max_iters):
          if debug_print:
              d_info = clr.OKGREEN + '\r[CG iteration: {}]'.format(i) + clr.ENDC
              sys.stdout.write(d_info)
              sys.stdout.flush()

          k = max(self.gap, i // self.gap)

          rn = self.ops['res_norm'](model)
          if rn < self.cg_num_err:
              break

          self.ops['cg_update'](model)
          dl_track.append(self.ops['dl'](model))

          if i > k:
              stop = (dl_track[i + 1] - dl_track[i + 1 - k]) / dl_track[i + 1]
              if not np.isnan(stop) and stop < 1e-4:
                  break

      if debug_print:
          sys.stdout.write('\n')
          sys.stdout.flush()

      if self.adjust_damping:
          dl = self.ops['dl'](model)

      self.ops['train'](model)

      if self.adjust_damping:
          loss_after_cg = self.loss(model.predict(model.inputs))
          reduction_ratio = (loss_after_cg - loss_before_cg) / dl

          if reduction_ratio < 0.25 and self.damping > self.damp_num_err:
              self.damping *= 1.5
          elif reduction_ratio > 0.75 and self.damping > self.damp_num_err:
              self.damping /= 1.5

  def __conjugate_gradient(self, gradients):
      with tf.name_scope('conjugate_gradient'):
          cg_update_ops = []

          prec = None
          if self.use_prec:
              if self.prec_loss is None:
                  self.prec_loss = tf.unstack(self.prec_loss)
                  batch_size = self.prec_loss.get_shape()[0]
              else:
                  self.prec_loss = [tf.gather(self.prec_loss, i)
                                    for i in range(self.batch_size)]
                  batch_size = len(self.prec_loss)
              prec = [[g ** 2 for g in tf.gradients(tf.gather(self.prec_loss, i),
                                                      self.W)] for i in range(batch_size)]
              prec = [(sum(tensor) + self.damping) ** (-0.75)
                      for tensor in np.transpose(np.array(prec))]

          Ax = None
          if self.use_gnm:
              Ax = self.__Gv(self.cg_delta)
          else:
              Ax = self.__Hv(gradients, self.cg_delta)

          b = [-grad for grad in gradients]
          bAx = [b - Ax for b, Ax in zip(b, Ax)]

          condition = tf.equal(self.cg_step, 0)
          r = [tf.where(condition, tf.assign(r, bax),
                        r) for r, bax in zip(self.residuals, bAx)]

          d = None
          if self.use_prec:
              d = [tf.where(condition, tf.assign(d, p * r),
                            d) for p, d, r in zip(prec, self.directions, r)]
          else:
              d = [tf.where(condition, tf.assign(d, r),
                            d) for d, r in zip(self.directions, r)]

          Ad = None
          if self.use_gnm:
              Ad = self.__Gv(d)
          else:
              Ad = self.__Hv(gradients, d)

          residual_norm = tf.reduce_sum([tf.reduce_sum(r ** 2) for r in r])

          alpha = tf.reduce_sum([tf.reduce_sum(d * ad) for d, ad in zip(d, Ad)])
          alpha = residual_norm / alpha

          if self.use_prec:
              beta = tf.reduce_sum([tf.reduce_sum(p * (r - alpha * ad) ** 2)
                                    for r, ad, p in zip(r, Ad, prec)])
          else:
              beta = tf.reduce_sum([tf.reduce_sum((r - alpha * ad) ** 2) for r, ad
                                    in zip(r, Ad)])

          self.beta = beta
          beta = beta / residual_norm

          for i, delta in reversed(list(enumerate(self.cg_delta))):
              update_delta = tf.assign(delta, delta + alpha * d[i],
                                        name='update_delta')
              update_residual = tf.assign(self.residuals[i], r[i] - alpha * Ad[i],
                                          name='update_residual')
              p = 1.0
              if self.use_prec:
                  p = prec[i]
              update_direction = tf.assign(self.directions[i],
                                            p * (r[i] - alpha * Ad[i]) + beta * d[i], name='update_direction')
              cg_update_ops.append(update_delta)
              cg_update_ops.append(update_residual)
              cg_update_ops.append(update_direction)

          with tf.control_dependencies(cg_update_ops):
              cg_update_ops.append(tf.compat.v1.assign_add(self.cg_step, 1))
          cg_op = tf.group(cg_update_ops)

      dl = tf.reduce_sum([tf.reduce_sum(0.5 * (delta * ax) + grad * delta)
                          for delta, grad, ax in zip(self.cg_delta, gradients, Ax)])

      return cg_op, residual_norm, dl

  def __Hv(self, grads, vec):
      with tf.GradientTape() as tape:
          grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
      Hv = tape.gradient(grad_v, self.W)
      Hv = [hv + self.damp_pl * v for hv, v in zip(Hv, vec)]

      return Hv

  def __Gv(self, vec):
      with tf.GradientTape() as tape:
          Jv = self.__Rop(self.model.output, self.W, vec)
          Jv = tf.reshape(tf.stack(Jv), [-1, 1])
          HJv = tape.gradient(tf.matmul(tf.transpose(tape.gradient(self.loss, self.model.output)[0]), Jv),
                              self.model.output, stop_gradients=Jv)[0]
      JHJv = tape.gradient(tf.matmul(tf.transpose(HJv), self.model.output), self.W,
                            stop_gradients=HJv)
      JHJv = [gv + self.damp_pl * v for gv, v in zip(JHJv, vec)]

      return JHJv

  def __Rop(self, f, x, vec):
      with tf.GradientTape() as tape:
          tape.watch(x)
          f = tf.gather(f, tf.range(self.batch_size))
      Rop = tape.gradient(f, x, output_gradients=vec)
      return Rop

  def __update_delta_0(self):
      update_delta_0_ops = []
      for delta in self.cg_delta:
          update_delta = tf.assign(delta, self.cg_decay * delta)
          update_delta_0_ops.append(update_delta)
      update_delta_0_op = tf.group(update_delta_0_ops)

      return update_delta_0_op

  def __train_op(self):
      update_ops = []
      delta_and_vars = list(zip(self.cg_delta, self.W))
      for delta, w in reversed(delta_and_vars):
          with tf.control_dependencies(update_ops):
              update_ops.append(tf.assign(w, w + self.learning_rate * delta))
      training_op = tf.group(update_ops)

      return training_op