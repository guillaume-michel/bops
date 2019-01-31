(in-package :bops2)

(declaim (inline array-row))
(defun array-row (arr index)
  "return the index row of the given array or the given array if its rank is 1"
  ;;(declare (type (array (unsigned-byte 64) (*)) arr))
  ;;(declare (type (unsigned-byte 64) index))
  (if (eq (array-rank arr) 1)
      arr
      (make-array (cdr (array-dimensions arr))
                  :element-type (array-element-type arr)
                  :displaced-to arr
                  :displaced-index-offset (* index
                                             (reduce #'*
                                                       (cdr (array-dimensions arr)))))))

(declaim (inline check-dimensions))
(defun check-dimensions (r x w)
  "r is a bit array of shape (B, M)
x is a bit array of shape (B, N)
w is a bit array of shape (M, N)"
  ;; check B
  (assert (eq (array-dimension r 0)
              (array-dimension x 0)))
  ;; check M
  (assert (eq (array-dimension r 1)
              (array-dimension w 0)))
  ;; check N
  (assert (eq (array-dimension x 1)
              (array-dimension w 1))))

(declaim (inline sign)
         (ftype (function (fixnum) fixnum) sign))
(defun sign (x)
  (declare (type fixnum x))
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (if (>= x 0)
      1
      0))

(declaim (inline dense1-elem)
         (ftype (function ((array (unsigned-byte 64) (*))
                           (array (unsigned-byte 64) (*)))
                          (unsigned-byte 64)) dense1-elem))
(defun dense1-elem (xi wj)
  "compute dense binary operations
xi is a uint64 array of shape (N/64)
wj is a uint64 array of shape (N/64)"
  (declare (optimize (speed 3) (debug 0) (safety 0))
           (type (array (unsigned-byte 64) (*)) xi wj))
  (let ((total 0)
        (N/64 (array-dimension wj 0)))
    (declare (type fixnum total N/64))
    (loop :for k :below N/64 :do
         (incf total (logcount (logxor (aref xi k)
                                       (aref wj k)))))
    (sign (- (the fixnum (* N/64 64))
             (* 2 total)))))

(declaim (inline dense1))
(defun dense1 (ri xi w)
  "Compute dense binary operations between x and w and store result in r
ri is a uint64 array of shape (M/64)
xi is a uint64 array of shape (N/64)
w is a uint64 array of shape (M, N/64)"
  (let ((M/64 (array-dimension ri 0)))
    (loop :for j1 :below M/64 :do
         (let ((res 0))
           (loop :for j2 :below 64 :do
                (let* ((wj (array-row w (+ (* j1 64)
                                           j2)))
                       (b (dense1-elem xi wj)))
                  (setf res (logior res (ash b j2)))))
           (setf (aref ri j1) res)))))

(declaim (inline dense))
(defun dense (r x w)
  "Compute dense binary operations between x and w and store result in r
r is a uint64 array of shape (B, M/64)
x is a uint64 array of shape (B, N/64)
w is a uint64 array of shape (M, N/64)"
  ;;(declare (optimize (speed 3) (debug 0) (safety 0)))
  ;;(check-dimensions r x w)
  (let ((B (array-dimension x 0)))
    (loop :for i :below B :do
         (let ((xi (array-row x i))
               (ri (array-row r i)))
           (dense1 ri xi w)))))

(defun test (&key (B 10) (N 256) (M 128))
  (declare (optimize (speed 3) (debug 0) (safety 0))
           (type fixnum B N M))
  (let* ((x (make-array `(,B ,(/ N 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (w (make-array `(,M ,(/ N 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (r (make-array `(,B ,(/ M 64)) :element-type '(unsigned-byte 64) :initial-element 0)))
    (time (dense r x w))
    (equalp r
            (make-array `(,B ,(/ M 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))))

(defun test2 (&key (B 10) (N (* 28 28 32)) (M (* 128 32)) (M2 (* 10 32)))
  (declare (optimize (speed 3) (debug 0) (safety 0))
           (type fixnum B N M M2))
  (let* ((x (make-array `(,B ,(/ N 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (w (make-array `(,M ,(/ N 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (r (make-array `(,B ,(/ M 64)) :element-type '(unsigned-byte 64) :initial-element 0))
         (w2 (make-array `(,M2 ,(/ M 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (r2 (make-array `(,B ,(/ M2 64)) :element-type '(unsigned-byte 64) :initial-element 0)))
    (time (progn (dense r x w)
                 (dense r2 r w2)))
    (and (equalp r
                 (make-array `(,B ,(/ M 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff))
         (equalp r2
                 (make-array `(,B ,(/ M2 64)) :element-type '(unsigned-byte 64) :initial-element #xffffffffffffffff)))))
