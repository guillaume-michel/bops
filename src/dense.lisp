(in-package #:bops)

(declaim (inline flatten))
(defun flatten (arr)
  (make-array (array-total-size arr)
              :element-type (array-element-type arr)
              :displaced-to arr))

(declaim (inline array-row))
(defun array-row (arr index)
  "return the index row of the given array or the given array if its rank is 1"
  ;;(declare (type (array bit (*)) arr))
  ;;(declare (type fixnum index))
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

(declaim (inline sign))
(defun sign (x)
  (declare (type fixnum x))
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (if (>= x 0)
      1
      0))

(declaim (inline dense1-elem))
(defun dense1-elem (xi wj)
  "compute dense binary operations
xi is a bit array of shape (N)
wj is a bit array of shape (N)"
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (let ((N (array-dimension wj 0)))
    (sign (- N (* 2 (reduce #'+ (bit-xor xi wj)))))))

(declaim (inline dense1))
(defun dense1 (ri xi w)
  "Compute dense binary operations between x and w and store result in r
ri is a bit array of shape (M)
xi is a bit array of shape (N)
w is a bit array of shape (M, N)"
  (let ((M (array-dimension w 0)))
    (loop :for j :below M :do
         (setf (aref ri j)
               (dense1-elem xi (array-row w j))))))

(declaim (inline dense))
(defun dense (r x w)
  "Compute dense binary operations between x and w and store result in r
r is a bit array of shape (B, M)
x is a bit array of shape (B, N)
w is a bit array of shape (M, N)"
  ;;(declare (optimize (speed 3) (debug 0) (safety 0)))
  (check-dimensions r x w)
  (let ((B (array-dimension x 0)))
    (loop :for i :below B :do
         (let ((xi (array-row x i))
               (ri (array-row r i)))
           (dense1 ri xi w)))))

(defun test (&key (B 10) (N 256) (M 128))
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (let* ((x (make-array `(,B ,N) :element-type 'bit :initial-element 1))
         (w (make-array `(,M ,N) :element-type 'bit :initial-element 1))
         (r (make-array `(,B ,M) :element-type 'bit :initial-element 0)))
    (time (dense r x w))
    (equalp r
            (make-array `(,B ,M) :element-type 'bit :initial-element 1))))

(defun test2 (&key (B 10) (N (* 28 28 32)) (M (* 128 32)) (M2 (* 10 32)))
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (let* ((x (make-array `(,B ,N) :element-type 'bit :initial-element 1))
         (w (make-array `(,M ,N) :element-type 'bit :initial-element 1))
         (r (make-array `(,B ,M) :element-type 'bit :initial-element 0))
         (w2 (make-array `(,M2 ,M) :element-type 'bit :initial-element 1))
         (r2 (make-array `(,B ,M2) :element-type 'bit :initial-element 0)))
    (time (progn (dense r x w)
                 (dense r2 r w2)))
    (and (equalp r
                 (make-array `(,B ,M) :element-type 'bit :initial-element 1))
         (equalp r2
                 (make-array `(,B ,M2) :element-type 'bit :initial-element 1)))))
