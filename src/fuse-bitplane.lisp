(in-package :bops)

(deftype fuse-bitplane-input (&optional N M B)
  `(simple-array bit (,N ,M ,B)))

(deftype fuse-bitplane-output (&optional N M)
  `(simple-array (unsigned-byte 8) (,N ,M)))

(defclass fuse-bitplane ()
  ()
  (:documentation "Fuse bitplane operator which should be the last dimension"))

(defmethod operator-output-shape ((operator fuse-bitplane) input-shape)
  (destructuring-bind (N M B) input-shape
    (declare (ignore B))
    `(,N ,M)))

(defmethod make-operator-output ((operator fuse-bitplane) input-shape)
  (make-array (operator-output-shape operator input-shape) :element-type '(unsigned-byte 8)))

(defmethod run-inference ((operator fuse-bitplane) (inputs list) (outputs list))
  (assert (and (not (null inputs))
               (not (null outputs))))
  (let ((input (car inputs))
        (output (car outputs)))
    (run-inference operator input output)))

(defmethod run-inference ((operator fuse-bitplane) input output)
  (assert (typep input 'fuse-bitplane-input))
  (assert (typep output 'fuse-bitplane-output))
  (assert (= (array-dimension input 0)
             (array-dimension output 0)))
  (assert (= (array-dimension input 1)
             (array-dimension output 1)))
  (assert (= (array-dimension input 2)
             8)
          ()
          "fuse-bitplane do not support bitplanes != 8")

  (fuse-bitplane-uint8 input output))
