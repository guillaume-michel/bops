(in-package #:bops)

(defun bit-and-vec (x y r size)
  (declare (type (unsigned-byte 32) size)
           (optimize (speed 3) (debug 0) (safety 0)))
  (sb-sys:with-pinned-objects (x y r)
    (let ((px (sb-sys:vector-sap x))
          (py (sb-sys:vector-sap y))
          (pr (sb-sys:vector-sap r)))
      (loop :for i :below size :by 16 :do
           (%bit-and i px py pr)))))

(declaim (inline sign))
(defun sign (x)
  (declare (type fixnum x))
  (declare (optimize (speed 3) (debug 0) (safety 0)))
  (if (>= x 0)
      1
      0))
