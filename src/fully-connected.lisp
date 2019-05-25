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
                   :type (integer 1 most-positive-fixnum) ; multiple of 64, 128, 256, 512 or 1024 for optimized implementations
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
   (transpose :initarg :transpose
              :type boolean
              :initform nil
              :reader binary-fc-transpose
              :documentation "if t y is transposed")
   (weights :type binary-fc-weights
            :accessor binary-fc-weights
            :documentation "W tensor in y = W*x+b")
   (biases :type binary-fc-biases
           :accessor binary-fc-biases
           :documentation "b tensor in y = W*x+b"))
  (:documentation "Binary fully connected operator"))

(defmethod initialize-instance :after ((operator binary-fully-connected) &key)
  (with-slots ((CHW input-neurones)
               (M output-neurones)
               (B bitplanes)) operator
    (setf (slot-value operator 'weights)
          (make-random-bit-vector `(,B ,M ,CHW)))
    (setf (slot-value operator 'biases)
          (make-random-bias-vector `(,B ,M)
                                   (+ CHW 1)))))

(defmethod print-object ((object binary-fully-connected) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (input-neurones output-neurones bitplanes) object
      (format stream ":bitplanes ~d :output-neurones ~d :input-neurones ~d" bitplanes output-neurones input-neurones))))

(defmethod operator-output-shape ((operator binary-fully-connected) input-shape)
  (with-slots ((M output-neurones)
               (B bitplanes)
               transpose) operator
    (let ((batch-size (car input-shape)))
      (if transpose
          `(,batch-size ,M ,B)
          `(,batch-size ,B ,M)))))

(defmethod make-operator-output ((operator binary-fully-connected) input-shape)
  (make-array (operator-output-shape operator input-shape) :element-type 'bit))

(defmethod run-inference ((operator binary-fully-connected) (inputs list) (outputs list))
  (assert (and (not (null inputs))
               (not (null outputs))))
  (let ((input (car inputs))
        (output (car outputs)))
    (run-inference operator input output)))

(defmethod run-inference ((operator binary-fully-connected) input output)
  (with-slots ((CHW input-neurones)
               (M output-neurones)
               (B bitplanes)
               transpose
               weights
               biases) operator

    (assert (typep input `(binary-fc-input * ,B ,CHW)))
    (if transpose
        (assert (typep output `(binary-fc-output * ,M ,B)))
        (assert (typep output `(binary-fc-output * ,B ,M))))
    (assert (= (array-dimension input 0)
               (array-dimension output 0)))

    (dense output weights input biases)))
