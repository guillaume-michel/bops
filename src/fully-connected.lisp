(in-package :bops)

(deftype negative-fixnum ()
  '(integer most-negative-fixnum 0))

(deftype binary-fc-weights (&optional B M CHW)
  `(simple-array bit (,B ,M ,CHW)))

(deftype binary-fc-biases (&optional B M)
  `(simple-array negative-fixnum (,B ,M)))

(deftype binary-fc-input (&optional N B CHW)
  `(simple-array bit (,N ,B ,CHW)))

(deftype binary-fc-output (&optional N B M)
  `(simple-array bit (,N ,B ,M)))

(defclass binary-fully-connected ()
  ((input-neurones :initarg :input-neurones
                   :type (and (integer 64 most-positive-fixnum)
                              (satisfies is-multiple-of-64))
                   :reader binary-fc-input-neurones
                   :documentation "number of input neurones")
   (output-neurones :initarg :output-neurones
                    :type (integer 1 most-positive-fixnum)
                    :reader binary-fc-output-neurones
                    :documentation "number of output neurones")
   (bitplanes :initarg :bitplanes
              :type (integer 1 most-positive-fixnum)
              :reader binary-fc-bitplanes
              :documentation "number of bitplanes")
   (weights :type binary-fc-weights
            :accessor binary-fc-weights
            :documentation "W tensor in y = W*x+b")
   (biases :type binary-fc-biases
           :accessor binary-fc-biases
           :documentation "b tensor in y = W*x+b"))
  (:documentation "Binary fully connected operator"))

(defmethod initialize-instance :after ((fc binary-fully-connected) &key)
  (with-slots ((CHW input-neurones)
               (M output-neurones)
               (B bitplanes)) fc
    (setf (slot-value fc 'weights)
          (make-random-bit-vector `(,B ,M ,CHW)))
    (setf (slot-value fc 'biases)
          (make-random-bias-vector `(,B ,M)
                                   (+ CHW 1)))))

(defmethod print-object ((object binary-fully-connected) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (input-neurones output-neurones bitplanes) object
      (format stream ":bitplanes ~d :output-neurones ~d :input-neurones ~d" bitplanes output-neurones input-neurones))))

(defmethod run-inference ((fc binary-fully-connected) (inputs list) (outputs list))
  (assert (and (not (null inputs))
               (not (null outputs))))
  (let ((input (car inputs))
        (output (car outputs)))
    (run-inference fc input output)))

(defmethod run-inference ((fc binary-fully-connected) input output)
  (with-slots ((CHW input-neurones)
               (M output-neurones)
               (B bitplanes)
               weights
               biases) fc

    (assert (typep input `(binary-fc-input * ,B ,CHW)))
    (assert (typep output `(binary-fc-output * ,B ,M)))
    (assert (= (array-dimension input 0)
               (array-dimension output 0)))

    (dense-v1 output weights input biases)))