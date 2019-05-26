(in-package :bops)

(defun uniform-bit-mutation (w pmut)
  (let ((result (make-random-bit-vector (array-dimensions w) :probability-one pmut)))
    (bit-xor w result result)
    result))

(defun uniform-fixnum-mutation (input pmut min max)
  (let* ((output (make-array (array-dimensions input) :element-type (array-element-type input)))
         (input-data (simple-array-vector input))
         (output-data (simple-array-vector output)))
    (iter (for i below (array-dimension input-data 0))
          (setf (aref output-data i)
                (if (<= (random 1.0f0) pmut)
                    (+ (random (- max min)) min)
                    (aref input-data i))))
    output))

(defclass uniform-mutation ()
  ((prob :initarg :prob
         :type single-float
         :accessor uniform-mutation-prob
         :documentation "uniform mutation probability"))
  (:documentation "uniform mutation strategy"))

(defmethod print-object ((object uniform-mutation) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (prob) object
      (format stream ":prob ~A" prob))))
