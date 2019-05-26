(in-package :bops)

(defun uniform-bit-crossover (w1 w2 pcross)
  (let* ((mask (make-random-bit-vector (array-dimensions w1) :probability-one pcross))
         (bitdiff (bit-and (bit-xor w1 w2)
                           mask)))
    (list
     (bit-xor w1 bitdiff)
     (bit-xor w2 bitdiff))))

(defun uniform-fixnum-crossover (input1 input2  pcross)
  (let* ((output1 (make-array (array-dimensions input1) :element-type (array-element-type input1)))
         (output2 (make-array (array-dimensions input2) :element-type (array-element-type input2)))
         (input1-data (simple-array-vector input1))
         (input2-data (simple-array-vector input2))
         (output1-data (simple-array-vector output1))
         (output2-data (simple-array-vector output2)))
    (iter (for i below (array-dimension input1-data 0))
          (if (<= (random 1.0f0) pcross)
              (progn
                (setf (aref output1-data i) (aref input2-data i))
                (setf (aref output2-data i) (aref input1-data i)))
              (progn
                (setf (aref output1-data i) (aref input1-data i))
                (setf (aref output2-data i) (aref input2-data i)))))
    (list output1 output2)))

(defclass uniform-crossover ()
  ((prob :initarg :prob
         :type single-float
         :accessor uniform-crossover-prob
         :documentation "uniform crossover probability"))
  (:documentation "uniform crossover strategy"))

(defmethod print-object ((object uniform-crossover) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (prob) object
      (format stream ":prob ~A" prob))))
