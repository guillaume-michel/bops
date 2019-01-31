(in-package #:bops)

(defknown (%bit-and) ((unsigned-byte 32)
                   system-area-pointer
                   system-area-pointer
                   system-area-pointer)
    (values)
    (any)
  :overwrite-fndb-silently t)

(define-vop (%bit-and)
  (:translate %bit-and)
  (:policy :fast-safe)
  (:args (index :scs (unsigned-reg))
         (vector1 :scs (sap-reg))
         (vector2 :scs (sap-reg))
         (result :scs (sap-reg)))
  (:arg-types unsigned-num
              system-area-pointer
              system-area-pointer
              system-area-pointer)
  (:results)
  (:temporary (:sc single-sse-reg) tmp)
  (:temporary (:sc unsigned-reg) idx)
  (:generator
   4
   (move idx index)
   (inst movaps
         tmp
         (make-ea :dword :base vector1 :disp 0 :index idx))
   (inst andps
         tmp
         (make-ea :dword :base vector2 :disp 0 :index idx))
   (inst movaps
         (make-ea :dword :base result :disp 0 :index idx)
         tmp)))

(define-vop (%bit-and/fast)
  (:translate %bit-and)
  (:policy :fast-safe)
  (:args (index :scs (unsigned-reg))
         (vector1 :scs (sap-reg))
         (vector2 :scs (sap-reg))
         (result :scs (sap-reg)))
  (:arg-types unsigned-num
              system-area-pointer
              system-area-pointer
              system-area-pointer)
  (:results)
  (:temporary (:sc single-sse-reg) tmp1)
  (:temporary (:sc single-sse-reg) tmp2)
  (:temporary (:sc unsigned-reg) idx)
  (:generator
   3
   (move idx index)
   (inst movaps
         tmp1
         (make-ea :dword :base vector1 :disp 0 :index idx))
   (inst movaps
         tmp2
         (make-ea :dword :base vector2 :disp 0 :index idx))
   (inst andps
         tmp1
         tmp2)
   (inst movaps
         (make-ea :dword :base result :disp 0 :index idx)
         tmp1)))

(defun %bit-and (i x y r)
  (declare (type (unsigned-byte 32) i)
           (type system-area-pointer r x y)
           (optimize (speed 3) (debug 0) (safety 0)))
  (%bit-and i x y r))

