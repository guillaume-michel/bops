(in-package :bops3)

(defknown %vec-xor ((unsigned-byte 32)
                    system-area-pointer
                    system-area-pointer
                    system-area-pointer)
    (values)
    (any)
  :overwrite-fndb-silently t)

(define-vop (%vec-xor)
  (:translate %vec-xor)
  (:policy :fast-safe)
  (:args (index :scs (unsigned-reg))
         (result :scs (sap-reg))
         (vector1 :scs (sap-reg))
         (vector2 :scs (sap-reg)))
  (:arg-types unsigned-num
              system-area-pointer
              system-area-pointer
              system-area-pointer)
  (:results)
  (:temporary (:sc unsigned-reg) tmp)
  (:generator
   4
   (inst mov
         tmp
         (make-ea :qword :base vector1 :disp 0 :index index))
   (inst xor
         tmp
         (make-ea :qword :base vector2 :disp 0 :index index))
   (inst mov
         (make-ea :qword :base result :disp 0 :index index)
         tmp)))

(defun %vec-xor (i r x y)
  (declare (type (unsigned-byte 32) i)
           (type system-area-pointer r x y)
           (optimize (speed 3) (debug 0) (safety 0)))
  (%vec-xor i r x y))

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
