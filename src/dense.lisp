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
                     (safety 0)
                     (debug 0)))

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

(defun dense-v1 (arr-y arr-w arr-x arr-b)
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
                     (safety 0)
                     (debug 0)))

  (sb-sys:with-pinned-objects (arr-w arr-x)
    (let* ((N (array-dimension arr-x 0))
           (B (array-dimension arr-x 1))
           (CHW (array-dimension arr-x 2))
           (M (array-dimension arr-w 1))
           (fx (aops:flatten arr-x))
           (fw (aops:flatten arr-w))
           (pw (sb-sys:vector-sap fw))
           (px (sb-sys:vector-sap fx)))

      (declare (type fixnum N B CHW M)
               (type system-area-pointer pw px))

      (iter (for iN below N)
            (declare (type fixnum iN))

            (iter (for iB below B)
                  (declare (type fixnum iB))

                  (iter (for iM below M)
                        (declare (type fixnum iM))

                        (setf (aref arr-y iN iB iM)
                              (sign (+ (aref arr-b iB iM)
                                       (let ((slice-x (+ (the fixnum (* iN CHW B))
                                                         (the fixnum (* iB CHW))))
                                             (slice-w (the fixnum (* iM CHW))))
                                         (declare (type fixnum slice-x slice-w))
                                         (iter (for iCHW below CHW by 256)
                                               (declare (type fixnum iCHW))
                                               (sum (+ (the fixnum (%popcnt (%xor-u64 (%sap-ref-u64 px (the fixnum (+ slice-x iCHW)))
                                                                                      (%sap-ref-u64 pw (the fixnum (+ slice-w iCHW))))))
                                                       (the fixnum (%popcnt (%xor-u64 (%sap-ref-u64 px (the fixnum (+ slice-x (+ iCHW 64))))
                                                                                      (%sap-ref-u64 pw (the fixnum (+ slice-w (+ iCHW 64)))))))
                                                       (the fixnum (%popcnt (%xor-u64 (%sap-ref-u64 px (the fixnum (+ slice-x (+ iCHW 128))))
                                                                                      (%sap-ref-u64 pw (the fixnum (+ slice-w (+ iCHW 128)))))))
                                                       (the fixnum (%popcnt (%xor-u64 (%sap-ref-u64 px (the fixnum (+ slice-x (+ iCHW 192))))
                                                                                      (%sap-ref-u64 pw (the fixnum (+ slice-w (+ iCHW 192)))))))) into acc)
                                               (declare (type fixnum acc))
                                               (finally (return acc)))))))))))))

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

(defun test-dense-v1 (&key
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
                (dense-v1 arr-y arr-w arr-x arr-b)))))

(defun check-dense-v1 (&key
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

    ;; warmup
    (dense-v1 arr-y1 arr-w arr-x arr-b)
    (dense-v1 arr-y2 arr-w arr-x arr-b)

    ;; bench
    (time (iter (repeat N-repeat)
                (dense-reference arr-y1 arr-w arr-x arr-b)))
    (time (iter (repeat N-repeat)
                (dense-v1 arr-y2 arr-w arr-x arr-b)))

    ;; check
    (equalp arr-y1 arr-y2)))
