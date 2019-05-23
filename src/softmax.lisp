(in-package :bops)

(deftype softmax-input (&optional N M)
  `(simple-array (unsigned-byte 8) (,N ,M)))

(deftype softmax-output (&optional N M)
  `(simple-array single-float (,N ,M)))

(defclass softmax ()
  ((theta :initarg :theta
          :type (integer 1 255)
          :initform 255
          :reader softmax-theta
          :documentation "nomalization factor for softmax"))
  (:documentation "Softmax operator"))

(defmethod print-object ((object softmax) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (theta) object
      (format stream ":theta ~d" theta))))

(defmethod run-inference ((operator softmax) (inputs list) (outputs list))
  (assert (and (not (null inputs))
               (not (null outputs))))
  (let ((input (car inputs))
        (output (car outputs)))
    (run-inference operator input output)))

(defmethod run-inference ((operator softmax) input output)
  (assert (typep input 'softmax-input))
  (assert (typep output 'softmax-output))
  (assert (equalp (array-dimensions input)
                  (array-dimensions outputs)))

  (with-slots (theta) operator
    (softmax input output :theta theta)))
