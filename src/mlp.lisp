(in-package :bops)

(defclass mlp ()
  ((scratchs :initarg :scratchs
             :initform nil
             :accessor mlp-scratchs
             :documentation "temporary buffers for mlp inference")
   (dims :initarg :dims
         :initform nil
         :accessor mlp-dims
         :documentation "list fo dimensions for the hidden layers")
   (bitplane :initarg :bitplane
             :initform 8
             :accessor mlp-bitplane
             :documentation "number of element in bitplane dimension")
   (batch-size :initarg :batch-size
               :initform nil
               :accessor mlp-batch-size
               :documentation "batch size used for mlp inference")))

(defmethod print-object ((object mlp) stream)
  (print-unreadable-object (object stream :type t :identity t)
    (with-slots (scratchs dims bitplane batch-size) object
      (format stream ":dims ~A :bitplane ~D :batch-size ~D" dims bitplane batch-size))))

(defun check-all-multiple-of (m l)
  (reduce (lambda (acc val)
            (and acc
                 (= (mod val m)
                    0)))
          l))

(defun mlp-check-dims (dims)
  (assert (>= (length dims)
              2))
  (check-all-multiple-of 64 (cdr (butlast dims))))

(defun make-mlp-weights (dims &key (B 8))
  (let ((results nil))
    (iter (for i below (- (length dims) 1))
          (push (make-random-bit-vector `(,B ,(nth (+ i 1) dims) ,(nth i dims)))
                results))
    (reverse results)))

(defun make-mlp-biases (dims &key (B 8))
  (let ((results nil))
    (iter (for i from 1 below (length dims))
          (push (make-random-bias-vector `(,B ,(nth i dims))
                                         (+ 2 (nth (- i 1) dims)))
                results))
    (reverse results)))

(defun make-mlp-scratchs (dims batch-size &key (B 8))
  (let ((results nil))
    (iter (for i from 1 below (length dims))
          (push (make-array `(,batch-size ,B ,(nth i dims)) :element-type 'bit)
                results))
    (reverse results)))

(defun make-mlp (dims batch-size &key (B 8))
  (mlp-check-dims dims)
  (make-instance 'mlp
                 :scratchs (make-mlp-scratchs dims batch-size :B B)
                 :dims dims
                 :bitplane B
                 :batch-size batch-size))

(defun mlp-run-inference (mlp input weights biases)
  (with-slots (scratchs dims bitplane batch-size) mlp
    (let ((inputs (cons input scratchs)))
      (iter (for i below (length weights))
            (let ((arr-y (nth i scratchs))
                  (arr-w (nth i weights))
                  (arr-x (nth i inputs))
                  (arr-b (nth i biases)))
              (dense-v1 arr-y arr-w arr-x arr-b))))
    (softmax-old (fuse-bitplane-uint8 (aops:permute '(0 2 1)
                                                    (car (last scratchs)))))))
