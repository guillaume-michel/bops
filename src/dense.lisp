(in-package :bops)

(defun vec-xor (N pr px py)
  (declare (type (unsigned-byte 64) N)
           (type system-area-pointer pr px py)
           (optimize (speed 3) (debug 0) (safety 0)))
  (loop :for i of-type fixnum :below (/ N 64) :by 1 :do
       (%vec-xor (* 8 i) pr px py)))

(defun test-xor (&key (N 256))
  (let* ((repeat-count 1000)
         (x1 (make-array N :element-type 'bit :initial-element 1))
         (x2 (make-array N :element-type 'bit :initial-element 0))
         (r (make-array N :element-type 'bit :initial-element 0)))
    (sb-sys:with-pinned-objects (r x1 x2)
      (let ((pr (sb-sys:vector-sap r))
            (px1 (sb-sys:vector-sap x1))
            (px2 (sb-sys:vector-sap x2)))
        (time (dotimes (count repeat-count) (vec-xor N pr px1 px2)))))
    (equalp r
            (make-array N :element-type 'bit :initial-element 1))))

(defun dense-reference-no-opt (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1)))

    (iter (for iN below N)
          (iter (for iB below B)
                (iter (for iM below M)
                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW)
                                           (sum (logxor (aref arr-x iN iB iCHW)
                                                        (aref arr-w iB iM iCHW))))))))))))

(defun dense-reference (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1)))

    (declare (type fixnum N B CHW M))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))
                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW)
                                           (declare (type fixnum iCHW))
                                           (sum (the fixnum (logxor (aref arr-x iN iB iCHW)
                                                                    (aref arr-w iB iM iCHW))) into acc)
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

;;; I need to write a macro for all of it
(defun dense (arr-y arr-w arr-x arr-b)
  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((CHW (array-dimension arr-x 2)))
    (cond ((= (mod CHW 1024) 0) (dense-1024 arr-y arr-w arr-x arr-b))
          ((= (mod CHW 512) 0) (dense-512 arr-y arr-w arr-x arr-b))
          ((= (mod CHW 256) 0) (dense-256 arr-y arr-w arr-x arr-b))
          ((= (mod CHW 128) 0) (dense-128 arr-y arr-w arr-x arr-b))
          ((= (mod CHW 64) 0) (dense-64 arr-y arr-w arr-x arr-b))
          (t (dense-reference arr-y arr-w arr-x arr-b)))))

(defun dense-64 (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1))
        (data-x (simple-array-vector arr-x))
        (data-w (simple-array-vector arr-w)))

    (declare (type fixnum N B CHW M)
             (type (simple-array bit (* * *)) arr-x arr-w))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))

                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW by 64)
                                           (declare (type fixnum iCHW))

                                           (let ((index-x (/ (array-row-major-index arr-x iN iB iCHW) 64))
                                                 (index-w (/ (array-row-major-index arr-w iB iM iCHW) 64)))
                                             (sum (+ (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 0))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 0))))) into acc))
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

(defun dense-128 (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1))
        (data-x (simple-array-vector arr-x))
        (data-w (simple-array-vector arr-w)))

    (declare (type fixnum N B CHW M)
             (type (simple-array bit (* * *)) arr-x arr-w))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))

                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW by 128)
                                           (declare (type fixnum iCHW))

                                           (let ((index-x (/ (array-row-major-index arr-x iN iB iCHW) 64))
                                                 (index-w (/ (array-row-major-index arr-w iB iM iCHW) 64)))
                                             (sum (+ (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 0))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 0))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 1))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 1))))) into acc))
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

(defun dense-256 (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1))
        (data-x (simple-array-vector arr-x))
        (data-w (simple-array-vector arr-w)))

    (declare (type fixnum N B CHW M)
             (type (simple-array bit (* * *)) arr-x arr-w))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))

                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW by 256)
                                           (declare (type fixnum iCHW))

                                           (let ((index-x (/ (array-row-major-index arr-x iN iB iCHW) 64))
                                                 (index-w (/ (array-row-major-index arr-w iB iM iCHW) 64)))
                                             (sum (+ (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 0))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 0))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 1))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 1))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 2))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 2))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 3))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 3))))) into acc))
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

(defun dense-512 (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1))
        (data-x (simple-array-vector arr-x))
        (data-w (simple-array-vector arr-w)))

    (declare (type fixnum N B CHW M)
             (type (simple-array bit (* * *)) arr-x arr-w))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))

                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW by 512)
                                           (declare (type fixnum iCHW))

                                           (let ((index-x (/ (array-row-major-index arr-x iN iB iCHW) 64))
                                                 (index-w (/ (array-row-major-index arr-w iB iM iCHW) 64)))
                                             (sum (+ (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 0))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 0))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 1))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 1))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 2))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 2))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 3))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 3))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 4))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 4))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 5))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 5))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 6))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 6))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 7))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 7))))) into acc))
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

(defun dense-1024 (arr-y arr-w arr-x arr-b)
  "Computes bitserial arr-y = arr-W * arr-x + arr-b

arr-y: array of shape (N, B, M) with type 'bit
arr-x: array of shape (N, B, CxHxW) with type 'bit
arr-w: array of shape (B, M, CxHxW) with type 'bit
arr-b: array of shape (B, M) with type (integer (- (+ CxHxW 1)) CxHxW))

where
 * N is the batch size dimension
 * B is the bitplane dimension
 * C is the channel dimension
 * H is the spatial height
 * W is the spatial width"

  (declare (type (simple-array bit (* * *)) arr-y arr-x arr-w)
           (type (simple-array fixnum (* *)) arr-b)
           (optimize (speed 3)
                     (compilation-speed 0)
                     (safety 3)
                     (debug 3)))

  (let ((N (array-dimension arr-x 0))
        (B (array-dimension arr-x 1))
        (CHW (array-dimension arr-x 2))
        (M (array-dimension arr-w 1))
        (data-x (simple-array-vector arr-x))
        (data-w (simple-array-vector arr-w)))

    (declare (type fixnum N B CHW M)
             (type (simple-array bit (* * *)) arr-x arr-w))

    (iter (for iN below N)
          (declare (type fixnum iN))

          (iter (for iB below B)
                (declare (type fixnum iB))

                (iter (for iM below M)
                      (declare (type fixnum iM))

                      (setf (aref arr-y iN iB iM)
                            (sign (+ (aref arr-b iB iM)
                                     (iter (for iCHW below CHW by 1024)
                                           (declare (type fixnum iCHW))

                                           (let ((index-x (/ (array-row-major-index arr-x iN iB iCHW) 64))
                                                 (index-w (/ (array-row-major-index arr-w iB iM iCHW) 64)))
                                             (sum (+ (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 0))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 0))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 1))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 1))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 2))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 2))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 3))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 3))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 4))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 4))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 5))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 5))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 6))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 6))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 7))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 7))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 8))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 8))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 9))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 9))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 10))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 10))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 11))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 11))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 12))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 12))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 13))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 13))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 14))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 14))))
                                                     (logcount (logxor (sb-kernel:%vector-raw-bits data-x (+ index-x 15))
                                                                       (sb-kernel:%vector-raw-bits data-w (+ index-w 15))))
                                                     ) into acc))
                                           (declare (type fixnum acc))
                                           (finally (return acc)))))))))))

(defun test-dense-reference-no-opt (&key
                                      (N-repeat 1)
                                      (N 1000)
                                      (B 8)
                                      (C 1)
                                      (H 32)
                                      (W H)
                                      (M 128))
  (let* ((arr-y (make-array `(,N ,B ,M) :element-type 'bit))
         (arr-x (make-array `(,N ,B ,(* C H W)) :element-type 'bit))
         (arr-w (make-array `(,B ,M ,(* C H W)) :element-type 'bit))
         (arr-b (make-array `(,B ,M) :element-type 'fixnum)))
    (time (iter (repeat N-repeat)
                (dense-reference-no-opt arr-y arr-w arr-x arr-b)))))

(defun test-dense-reference (&key
                               (N-repeat 1)
                               (N 1000)
                               (B 8)
                               (C 1)
                               (H 32)
                               (W H)
                               (M 128))
  (let* ((arr-y (make-array `(,N ,B ,M) :element-type 'bit))
         (arr-x (make-array `(,N ,B ,(* C H W)) :element-type 'bit))
         (arr-w (make-array `(,B ,M ,(* C H W)) :element-type 'bit))
         (arr-b (make-array `(,B ,M) :element-type 'fixnum)))
    (time (iter (repeat N-repeat)
                (dense-reference arr-y arr-w arr-x arr-b)))))

(defun test-dense (&key
                     (N-repeat 1)
                     (N 1000)
                     (B 8)
                     (C 1)
                     (H 32)
                     (W H)
                     (M 128))
  (let* ((arr-y (make-array `(,N ,B ,M) :element-type 'bit))
         (arr-x (make-array `(,N ,B ,(* C H W)) :element-type 'bit))
         (arr-w (make-array `(,B ,M ,(* C H W)) :element-type 'bit))
         (arr-b (make-array `(,B ,M) :element-type 'fixnum)))
    (time (iter (repeat N-repeat)
                (dense arr-y arr-w arr-x arr-b)))))

(defun check-dense (&key
                      (N-repeat 1)
                      (N 1000)
                      (B 8)
                      (C 1)
                      (H 32)
                      (W H)
                      (M 128))
  (let* ((arr-y1 (make-array `(,N ,B ,M) :element-type 'bit))
         (arr-y2 (make-array `(,N ,B ,M) :element-type 'bit))
         (arr-x (make-array `(,N ,B ,(* C H W)) :element-type 'bit))
         (arr-w (make-array `(,B ,M ,(* C H W)) :element-type 'bit))
         (arr-b (make-array `(,B ,M) :element-type 'fixnum)))

    (dense-reference arr-y1 arr-w arr-x arr-b)

    ;; warmup
    (dense arr-y2 arr-w arr-x arr-b)
    (dense arr-y2 arr-w arr-x arr-b)

    ;; bench
    (time (iter (repeat N-repeat)
                (dense arr-y2 arr-w arr-x arr-b)))

    ;; check
    (equalp arr-y1 arr-y2)))
