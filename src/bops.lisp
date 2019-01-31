(in-package #:bops)

(defun bit-and-vec (x y r size)
  (declare (type (unsigned-byte 32) size)
           (optimize (speed 3) (debug 0) (safety 0)))
  (sb-sys:with-pinned-objects (x y r)
    (loop :for i :below size :by 16 :do
         (%bit-and i
                   (sb-sys:vector-sap x)
                   (sb-sys:vector-sap y)
                   (sb-sys:vector-sap r)))))
