(in-package :bops)

(defun softmax-denom (arr)
  (declare (type (simple-array (unsigned-byte 8) (*)) arr)
           (optimize (speed 3) (debug 3) (safety 3)))
  (iter (for i below (array-dimension arr 0))
        (declare (type fixnum i))
        (sum (exp (/ (aref arr i) 255f0)) into acc)
        (declare (type single-float acc))
        (finally (return acc))))

(defun softmax (arr)
  (declare (type (simple-array (unsigned-byte 8) (* *)) arr)
           (optimize (speed 3) (debug 0) (safety 0)))
  (let* ((N (array-dimension arr 0))
         (M (array-dimension arr 1))
         (results (make-array `(,N ,M) :element-type 'single-float)))
    (declare (type fixnum N M)
             (type (simple-array single-float (* *)) results))
    (iter (for i below N)
          (declare (type fixnum i))
          (let ((slice (cl-slice:slice arr i t)))
            (declare (type (simple-array (unsigned-byte 8) (*)) slice))
            (iter (for j below M)
                  (declare (type fixnum j))
                  (setf (aref results i j) (/ (the float (exp (/ (aref slice j) 255f0)))
                                              (the float (softmax-denom slice)))))))
    results))

(defun argmax-helper (arr)
  (declare (type (simple-array single-float (*)) arr)
           (optimize (speed 3) (debug 3) (safety 3)))
  (let ((best-index -1)
        (best-value most-negative-single-float))
    (iter (for i below (array-dimension arr 0))
          (declare (type fixnum i))
          (if (> (aref arr i)
                 best-value)
              (progn
                (setf best-index i)
                (setf best-value (aref arr i)))))
    best-index))

(defun argmax (arr)
  (declare (type (simple-array single-float (* *)) arr)
           (optimize (speed 3) (debug 0) (safety 0)))
  (let* ((N (array-dimension arr 0))
         (results (make-array N :element-type 'fixnum)))
    (iter (for i below N)
          (declare (type fixnum i))
          (let ((slice (cl-slice:slice arr i t)))
            (setf (aref results i) (argmax-helper slice))))
    results))

(defun accuracy (y yhat)
  (assert (= (array-total-size y)
             (array-total-size yhat)))
  (let ((accuracy 0))
    (iter (for i below (array-total-size y))
          (if (= (aref y i)
                 (aref yhat i))
              (incf accuracy)))
    (float (/ accuracy (array-total-size y)))))

(defun loss (y yhat)
  "y is output of softmax.
yhat are ground truth labels"
  (declare (type (simple-array single-float (* *)) y)
           (type (simple-array (unsigned-byte 8) (*)) yhat)
           (optimize (speed 3) (debug 3) (safety 3)))

  (let ((N (array-dimension y 0)))
    (iter (for i below N)
          (declare (type fixnum i))
          (sum (- 1 (aref y i (aref yhat i))) into acc)
          (declare (type single-float acc))
          (finally (return (float (/ acc N)))))))

#|

(destructuring-bind (x-train y-train x-test y-test) (prepare-mnist (load-mnist))
    (print (array-dimensions x-train))
    (print (array-dimensions y-train))
    (print (array-dimensions x-test))
    (print (array-dimensions y-test)))

(defparameter datas (prepare-mnist (load-mnist)))
(defparameter arr-x (first datas))
(defparameter arr-w (make-random-bit-vector '(8 10 1024)))
(defparameter arr-b (make-random-bias-vector '(8 10) 125))
(defparameter arr-y (make-array '(60000 8 10) :element-type 'bit))

(dense-v1 arr-y arr-w arr-x arr-b)

(defparameter res (fuse-bitplane-uint8 (aops:permute '(0 2 1) arr-y)))


(defparameter train-labels (second datas))

|#
