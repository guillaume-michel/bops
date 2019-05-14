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

;;-----------------------------------------------------------

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

;;-----------------------------------------------------------

(defknown %sap-ref-u64 (system-area-pointer
                        (unsigned-byte 64))
    (unsigned-byte 64)
    (sb-c:foldable sb-c:flushable sb-c:movable)
  :overwrite-fndb-silently t)

(define-vop (%sap-ref-u64)
  (:translate %sap-ref-u64)
  (:policy :fast-safe)
  (:args (data :scs (sap-reg))
         (index :scs (unsigned-reg)))
  (:arg-types system-area-pointer
              unsigned-num)
  (:results (r :scs (unsigned-reg)))
  (:result-types unsigned-num)
  (:generator
   1
   (inst mov
         r
         (make-ea :qword :base data :disp 0 :index index))))

(defun %sap-ref-u64 (data index)
  (%sap-ref-u64 data index))

;;-----------------------------------------------------------

(defknown %xor-u64 ((unsigned-byte 64)
                    (unsigned-byte 64))
    (unsigned-byte 64)
    (sb-c:foldable sb-c:flushable sb-c:movable)
  :overwrite-fndb-silently t)

(define-vop (%xor-u64)
  (:translate %xor-u64)
  (:policy :fast-safe)
  (:args (x :scs (unsigned-reg) :target r)
         (y :scs (unsigned-reg)))
  (:arg-types unsigned-num
              unsigned-num)
  (:results (r :scs (unsigned-reg)))
  (:result-types unsigned-num)
  (:generator
   1
   (move r x)
   (inst xor r y)))

(defun %xor-u64 (x y)
  (%xor-u64 x y))

;;-----------------------------------------------------------

(defknown %popcnt ((unsigned-byte 64))
    (integer 0 64)
    (sb-c:foldable sb-c:flushable sb-c:movable)
  :overwrite-fndb-silently t)

(define-vop (%popcnt)
  (:policy :fast-safe)
  (:translate %popcnt)
  (:args (x :scs (unsigned-reg) :target r))
  (:arg-types unsigned-num)
  (:results (r :scs (unsigned-reg)))
  (:result-types unsigned-num)
  (:generator 3
    (unless (location= r x) ; only break the spurious dep. chain
      (inst xor r r))       ; if r isn't the same register as x.
    (inst popcnt r x)))

(defun %popcnt (x)
  (%popcnt x))

;;-----------------------------------------------------------

(defknown %add-u64 ((unsigned-byte 64)
                    (unsigned-byte 64))
    (unsigned-byte 64)
    (sb-c:foldable sb-c:flushable sb-c:movable)
  :overwrite-fndb-silently t)

(define-vop (%add-u64)
  (:translate %add-u64)
  (:policy :fast-safe)
  (:args (x :scs (unsigned-reg) :target r)
         (y :scs (unsigned-reg)))
  (:arg-types unsigned-num
              unsigned-num)
  (:results (r :scs (unsigned-reg)))
  (:result-types unsigned-num)
  (:generator
   1
   (move r x)
   (inst add r y)))

(defun %add-u64 (x y)
  (%add-u64 x y))

;;-----------------------------------------------------------

#|

(defparameter *a1* #*11110000111100001111000011110000111100001111000011110000111100001111000011110000111100001111000011110000111100001111000011110000)

(defparameter *a2* #*01010101010111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)

(defun f (px py i)
  (declare (type (unsigned-byte 64) i)
           (type system-area-pointer px py)
           (optimize (speed 3) (debug 0) (safety 0)))

  (let ((xi (%sap-ref-u64 px i))
        (yi (%sap-ref-u64 py i)))
    (%popcnt (%xor-u64 xi yi))))

(defun f2 (px py i)
  (declare (type (unsigned-byte 64) i)
           (type system-area-pointer px py)
           (optimize (speed 3) (debug 0) (safety 0)))

  (let* ((next-i (%add-u64 i 8))
         (xi (%sap-ref-u64 px i))
         (yi (%sap-ref-u64 py i))
         (xip1 (%sap-ref-u64 px next-i))
         (yip1 (%sap-ref-u64 py next-i))
         (xi-xor-yi (%xor-u64 xi yi))
         (xip1-xor-yip1 (%xor-u64 xip1 yip1)))
    (%add-u64 (%popcnt xi-xor-yi)
       (%popcnt xip1-xor-yip1))))

(defun f3 (px py i)
  (declare (type (unsigned-byte 32) i)
           (type system-area-pointer px py)
           (optimize (speed 3) (debug 0) (safety 0)))

  (let* ((next-i (+ i 8))
         (xi (%sap-ref-u64 px i))
         (yi (%sap-ref-u64 py i))
         (xip1 (%sap-ref-u64 px next-i))
         (yip1 (%sap-ref-u64 py next-i))
         (xi-xor-yi (%xor-u64 xi yi))
         (xip1-xor-yip1 (%xor-u64 xip1 yip1)))
    (%add-u64 (%popcnt xi-xor-yi)
       (%popcnt xip1-xor-yip1))))

(defun f4 (px py N)
  (declare (type fixnum N)
           (type system-area-pointer px py)
           (optimize (speed 3) (debug 0) (safety 0)))

  (loop :for i of-type fixnum :below (/ N 8) :by 8 :sum
         (%popcnt (%xor-u64 (%sap-ref-u64 px i)
                            (%sap-ref-u64 py i))) fixnum))

(let ((p1 (sb-sys:vector-sap *a1*))
      (p2 (sb-sys:vector-sap *a2*)))
  (f4 p1 p2 (array-dimension *a1* 0)))

(defun random-bit-vector (n)
  (make-array n
              :element-type 'bit
              :initial-contents (loop :repeat n
                                   :collect (random 2))))

(defun bench-xor-popcnt (&key (n (* 8 1024 1024 1)) (m 1000))
  (let ((a1 (random-bit-vector n))
        (a2 (random-bit-vector n)))
    (sb-sys:with-pinned-objects (a1 a2)
      (let ((pa1 (sb-sys:vector-sap a1))
            (pa2 (sb-sys:vector-sap a2)))
        (time (dotimes (i m)
                (f4 pa1 pa2 (array-dimension a1 0))))))))

|#
